# Rutina que preprocesa y transforma los datos para series de tiempo
# Roberto Gonzalez
# Pedro Fluxa
# ------------------------------------------------------------------
# Entrada: 2 o mas archivos .csv asincronos.
# Salida: Archivo binario hdf5 con chunks de datos sincronizados 
#
# Cada archivo csv debe tener una columna temporal, con un sampling 
# que puede ser arbitrario y asincrono respecto a los otros archivos
# Pueden haber gaps en los datos 
# 
# El parametro de entrada es un archivo json del tipo:
# config/sma_uai_alldata.json
# Este archivo contiene informacion de los archivos de entrada/salida,
# las columnas y variables de conversion.
#
# El programa toma los archivos y busca bloques continuos de datos
# en donde ninguna columna tenga un gap mayor a
# "max_delta_between_blocks_sec" segundos, si es menor
# los datos se interpolaran
# Cada bloque continuo de datos o "chunk" debe tener una duracion
# minima de "min_chunk_duration_sec" segundos o es descartado
# Luego los datos son sincronizados con un sampling uniforme
# dado por "sample_spacing"
#
# como resultado en el archivo hdf5 se esribiran multiples chunks
# de datos continuos, si "min_chunk_duration_sec" es muy grande
# y se genera 1 o muy pocos chunks, no es conveniente ya que 
# para construir el train/testing set deben haber suficientes
# chunks para asignarse a cada set. i.e. 20 chunks iguales
# implica un split de train/testing con un error del 5% al menos
# respecto al ratio definido.

import os
import sys
import json
import pandas
import numpy

# busca bloques continuos de tiempo donde no ocurra una separacion
# de tiempo mayor que maxStep y donde el bloque tenga un largo al 
# menos de minDuration.
def get_cont_chunks(df, date_col, maxStep, minDuration):
    timeArray = df[date_col]
    # indexes where a discontinuity in time occurs
    #ojo que funciona solo para numpy 1.20+
    idxs = numpy.where(numpy.diff(timeArray) > maxStep)[0]
    if len(idxs) == 0:
        # trick: add a last time sample with a huge jump to make the routine
        # consider a contiguous block as a single block of data
        timeArray_hack = numpy.concatenate([timeArray, 1e45])
        numpy.where(numpy.diff(timeArray_hack) > maxStep)[0]
        return [0, timeArray.size]
    print("found {} discontinuities".format(len(idxs)))
   
    leftIdx = 0
    rightIdx = -1
    interval_list = list()
    for idx in idxs:
        rightIdx = idx
        duration = timeArray[rightIdx] - timeArray[leftIdx]
        if duration > minDuration:
            interv = pandas.Interval(timeArray[leftIdx], timeArray[rightIdx])
            interval_list.append(interv)
        leftIdx = rightIdx + 1
    intervals = pandas.arrays.IntervalArray(interval_list)
    return intervals

if __name__ == '__main__':
    
    # Lee el archivo json con la configuracion de los datos
    config_path = sys.argv[1]
    
    # Mismas rutinas de carga ocupadas a lo largo del codigo.
    config_file = open(config_path, 'r')
    ds_config = json.loads(config_file.read())
    data_config = ds_config['dataset']
    config_file.close()
    chunk_config = data_config.get("chunk_config")
    training_config = data_config.get("training_config")
    source_configs = data_config.get("data_sources")
    hdf5_path = os.path.join(chunk_config['path_to_hdf5'], chunk_config['tag'])
    sample_spacing_min = chunk_config['sample_spacing']
    min_chunk_duration_sec = chunk_config["min_chunk_duration_sec"]
    max_sync_block_dt_sec = chunk_config["max_delta_between_blocks_sec"]
    min_date = pandas.to_datetime(chunk_config.get('min_date', "1900-01-01T00:00:00"))
    max_date = pandas.to_datetime(chunk_config.get('max_date', "2100-12-31T00:00:00"))

    print("deleting ",hdf5_path)
    try:
        os.remove(hdf5_path)
        print("file deleted")
    except:
        print("file does not exist")

    
    dfs = list()
    cont_date_intervals = list()
    for source_name, source_info in source_configs.items():
        print("[INFO] loading source {}".format(source_name))
        fpath = source_info['file']
        field_list = source_info['field_list']
        cos_sin_fields = source_info.get('cos_sin_fields', None)
        date_col = source_info.get('date_time_column_name', 'Date time')
        date_fmt = source_info.get('date_time_column_format', "%Y-%m-%dT%H:%M:%S")
        data = pandas.read_csv(fpath)
        data = data.dropna()
        data = data.reset_index(drop=True)
        data['master_datetime'] = pandas.to_datetime(data[date_col], format=date_fmt)
        data = data.drop([date_col], axis=1)
        # Genera cos y sin de la hora del dia.
        if cos_sin_fields is not None:
            for func, pname in zip([numpy.cos, numpy.sin], ["Cosine", "Sine"]):
                for fname in cos_sin_fields:
                    new_fname = pname + " " + fname
                    field_data = data[fname]
                    if 'deg' in fname:
                        field_data = field_data*numpy.pi/180.0
                    data[new_fname] = field_data
        # generamos los chunks continuos de datos.
        chunks = get_cont_chunks(
            data, 'master_datetime',
            pandas.Timedelta(max_sync_block_dt_sec, unit='s'), 
            pandas.Timedelta(min_chunk_duration_sec, unit='s'))
        dfs.append(data)
        cont_date_intervals.append(chunks)
        print(chunks)
    
    # sincronizamos y escribimos cada chunk
    hdfs = pandas.HDFStore(hdf5_path)
    ik = 0 
    n_samples = 0
    data_overlaps = list()
    intervals_i = cont_date_intervals[0]
    for inter_i in intervals_i:
        n_source_overlaps = 0
        interval_overlaps = list()
        # check overlap
        for intervals_j in cont_date_intervals[1::]:
            source_overlaps = list()
            overlap_mask = intervals_j.overlaps(inter_i)
            overlap_inter = intervals_j[overlap_mask]
            if len(overlap_inter) > 0:
                n_source_overlaps += 1 
                # find overlaps
                for overlap in overlap_inter:
                    o_left = max(inter_i.left, overlap.left)
                    o_right = min(inter_i.right, overlap.right)
                    source_overlaps.append(pandas.Interval(o_left, o_right, closed='neither')) #closed='neither' orig code
            interval_overlaps.append(source_overlaps)
        interval_overlaps.append(source_overlaps)
        # si es verdadero hay overlap entre todas las fuentes
        if n_source_overlaps != len(source_configs.keys()) - 1:
            continue
        # construye el dataframe con todas las columnas que coinciden
        resample_ok = True
        all_slices = list()
        for src_idx, src_os in enumerate(interval_overlaps):
            df = dfs[src_idx]
            src_o = src_os[0]
            odf = df[(df['master_datetime'] >= src_o.left) & (df['master_datetime'] <= src_o.right)] #seleccionamos el chunk
            odf = odf.set_index(pandas.DatetimeIndex(odf['master_datetime']))
            odf = odf.drop(['master_datetime'], axis=1)
            odf = odf.dropna()
            try:
                df_slice = odf.resample(sample_spacing_min, closed=None).bfill()
                all_slices.append(df_slice)
            except:
                resample_ok = False
                print("fail")
                break
        if resample_ok:
            synced_df = pandas.concat(all_slices, axis=1, join='inner')
            if len(synced_df) > 1:
                #  cos/sin day
                datetime = synced_df.index.to_series()
                if datetime.iloc[0] < min_date:
                    continue
                if datetime.iloc[-1] > max_date:
                    continue
                print(datetime.iloc[0], "to", datetime.iloc[-1])
                sec_day = (datetime - datetime.dt.normalize())/pandas.Timedelta(seconds=1)
                cos_sec_day = numpy.cos(2*numpy.pi*sec_day.values/86400.0)
                sin_sec_day = numpy.sin(2*numpy.pi*sec_day.values/86400.0)
                synced_df['Cosine Day'] = cos_sec_day
                synced_df['Sine Day'] = sin_sec_day
                #print(synced_df.columns)
                synced_df.to_hdf(hdfs, "chunk_{:d}".format(ik), format='table')
                n_samples += len(synced_df)
                print('chunk=',ik,' nsamples=',n_samples)
                
        else:
            print("resampled failed, ignoring chunk")
        ik += 1
