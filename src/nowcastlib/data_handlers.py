#funciones de manejo y sincronizacion de datos.
import pandas
import numpy


#clase genérica de manejo de datos
class BaseHandler(object):
    '''
    '''
    _hdf_store = None
    _df_list = list()
    _trainval_frac = -1.0
    _x_col_names = list()
    _y_col_names = list()
    
    _block_stride = 1
    _lag = None
    _boost = None
    
    _has_normalization = False
    _norm_dict = dict()
   
    _allowed_dyn_range = (-1e20, 1e20)
    _allowed_std_range = (-1e20, 1e20)
    
    _smooth_steps = 0
    # basehandler support up to three inputs and three outputs
    _n_inputs = 0
    _n_outputs = 0
    
    _train_x = list()
    _train_y = list()
    
    _test_x = list()
    _test_y = list()
    _testing_keys = list()
    _training_keys = list()
    _val_x = list()
    _val_y = list()
    _boost_data = list()
    
    def __init__(self, path_to_hdf, x_col_names, name_of_y_col, max_df):
        '''
        '''
        self._hdf_store = pandas.HDFStore(path_to_hdf)
        df_list = self._hdf_store.keys()
        self._hdf_store.close()
        
        self._df_list = df_list[0:max_df]
        self._x_col_names = x_col_names
        # pseudo-cast to a list to have consistent array shapes 
        self._y_col_names = [name_of_y_col,]
    
    def set_keys(self, keys):
        '''
        '''
        self._df_list = keys
    
    def set_smoothing(self, s):
        '''
        '''
        self._smooth_steps = s

    def smooth(self, x):
        '''convolves x with a rectangular window of width smooth_steps
        '''
        window_len = self._smooth_steps
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
        w = numpy.ones(window_len, 'd')
        xp = numpy.concatenate([w*x[0], x, w*x[-1]])
        xc = numpy.convolve(w/w.sum(), xp, mode='same')
        y = xc[window_len:-window_len]
        return y

    def set_stride(self, block_stride):
        '''
        '''
        self._block_stride = block_stride
    
    def set_trainval_frac(self, f):
        '''
        '''
        if f < 0 or f > 1.0:
            raise ValueError("training/val fraction must be between 0 and 1.")
        self._trainval_frac = f
    
    def set_allowed_dyn_range(self, minr, maxr):
        '''
        '''
        self._allowed_dyn_range = (minr, maxr)
    
    def set_allowed_std_range(self, min_std, max_std):
        '''
        '''
        self._allowed_std_range = (min_std, max_std)
    
    def set_lag(self, l):
        '''
        '''
        self._lag = int(l)
        
    def set_boost(self, b):
        '''
        '''
        self._boost = int(b)
    
    def set_normalization(self, norm_dict):
        '''
        '''
        self._norm_dict = norm_dict.copy()
        self._has_normalization = True 
          
    def get_training_keys(self):
        '''
        '''
        return self._training_keys

    def get_testing_keys(self):
        '''
        '''
        return self._testing_keys

    def get_training_set(self):
        '''
        '''
        return self._train_x, self._train_y
        
    def get_testing_set(self):
        '''
        '''
        return self._test_x, self._test_y

    def check_dataframe_dynamic_range(self, df):
        '''
        '''
        if df.min().any() < self._allowed_dyn_range[0]:
            return False
        if df.max().any() > self._allowed_dyn_range[1]:
            return False
        return True
    
    def check_block_variance(self, block_data):
        '''
        '''
        std_data = numpy.std(block_data)
        if std_data < self._allowed_std_range[0]:
            return True
        if std_data > self._allowed_std_range[1]:
            return True
        return False
    
    #En la configuracion de training las variables donde se define el par de normalizacion, se normalizaran segun esto.
    def normalize_dataframe(self, df):
        '''
        '''
        if not self._has_normalization:
            raise RuntimeError("normalization is not set.")
        for fname, norm_range in self._norm_dict.items():
            fmin = norm_range[0]
            fmax = norm_range[1]
            fscl = fmax - fmin
            df[fname] = (df[fname] - fmin)/fscl
        return df

    #funcion principal de sincronizacion y creacion de chunks
    def blocks_from_dataframe(self, 
        dataframe, 
        filter_by_std, only_out_of_range):
        '''
        '''
        # block status = True means it is OK
        bad_block = False
        # build lag/boost blocks
        lag_blocks_x = list()
        lag_blocks_y = list()
        boost_blocks = list()
        
        n_samp = dataframe.shape[0]
        # check data chunk is long enough to build at least a single block
        if n_samp < self._lag + self._boost:
            bad_block = True
            return bad_block, None, None, None
        # normalize
        if self._has_normalization:
            dataframe = self.normalize_dataframe(dataframe)
        # look for dynamic range
        in_range = self.check_dataframe_dynamic_range(dataframe)
        if not in_range:
            bad_block = True
            return bad_block, None, None, None
        # use numpy arrays as it is sligthly faster
        x_data = dataframe[self._x_col_names].values
        y_data = dataframe[self._y_col_names].values
        if self._smooth_steps > 0:
            y_temp = y_data.ravel()
            y_temp = self.smooth(y_temp)
            y_data = y_temp.reshape((-1,1))

        ix = 0
        iy = self._lag
        while iy + self._boost < n_samp:
            lag_data_x = x_data[ix:ix + self._lag, :]
            lag_data_y = y_data[ix:ix + self._lag, :]
            boost_data = y_data[iy:iy + self._boost, :]
            # if boost_block has suficcient data, we reached the end
            if len(boost_data) < self._boost:
                ix += self._block_stride
                iy += self._block_stride
                break
            # filter by std if required
            # Esto basicamente elimina el chunk donde la varianza de la prediccion normalizada 
            # se escapa del rango definido por el param "normalized_target_variance_range":
            out_of_range = False
            if filter_by_std:
                out_of_range = self.check_block_variance(boost_data)
                if only_out_of_range:
                    out_of_range = not out_of_range
            if out_of_range:
                ix += self._block_stride
                iy += self._block_stride
                continue

            lag_blocks_x.append(lag_data_x)
            lag_blocks_y.append(lag_data_y)
            boost_blocks.append(boost_data)
            
            ix += self._block_stride
            iy += self._block_stride
        
        lag_blocks_x = numpy.asarray(lag_blocks_x, dtype='float32')
        lag_blocks_y = numpy.asarray(lag_blocks_y, dtype='float32')
        boost_blocks = numpy.asarray(boost_blocks, dtype='float32')
        
        return bad_block, lag_blocks_x, lag_blocks_y, boost_blocks


# Clase para generar el output de convertir la variable de prediccion en clases categoricas.
class MaxCategorical(BaseHandler):
    '''
    '''
    def __init__(self, path_to_hdf, x_col_names, name_of_y_col, max_df=-1):
        '''
        '''
        super().__init__(path_to_hdf, x_col_names, name_of_y_col, max_df)
        self.n_max_avg = 0
    
    def set_boost_avg_steps(self, n):
        '''
        '''
        self.n_max_avg = n

    def set_ranges(self, ranges):
        '''
        '''
        self.n_ranges = len(ranges)
        self.ranges = ranges + [1e10,]

    # convierte el boost de la prediccion en las variables categoricas definidas por las clases y rangos.
    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        rngs = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
 
            # En esta parte se seleccionan los ultimos 8 boost para la prediccion
            ii = len(boost_data)-8
            if ii<0:
                ii=0

            # aqui se calcula el maximo valor de los ultimos 8 elementos
            max_avg = -1e10
            while ii + self.n_max_avg < len(boost_data):
                yw = boost_data[ii:ii+self.n_max_avg, :].ravel()
                local_avg = numpy.max(yw) #from ave to max ROB
                if local_avg > max_avg:
                    max_avg = local_avg
                ii += 1
            # truncate negative values
            if max_avg < 0:
                max_avg = 1e-5
            ranges = numpy.zeros(self.n_ranges, dtype='float32')                
            for i_r in range(self.n_ranges):
                lr = self.ranges[i_r]
                rr = self.ranges[i_r+1]
                if max_avg >= lr and max_avg < rr:
                    ranges[i_r] = 1.0
            ranges_ok = numpy.sum(ranges) == 1.0
            # this should never happen
            if not ranges_ok:
                raise RuntimeError("bad range output")
            rngs.append(ranges)
        rngs = numpy.asarray(rngs, dtype='float32')        
        return rngs

    # rutina principal de generacion del dataset
    def build_dataset(self,
        filter_by_std=False, only_out_of_range=False):
        '''
        '''
        if self._trainval_frac < 0 or self._trainval_frac > 1.0:
            raise ValueError("training/val fraction must be in the ]0,1[ range")

        shape_x1 = (1, self._lag, len(self._x_col_names))
        shape_y1 = (1, self.n_ranges)
        
        test_x1 = numpy.zeros(shape_x1)
        test_y1 = numpy.zeros(shape_y1)
        
        train_x1 = numpy.zeros(shape_x1)
        train_y1 = numpy.zeros(shape_y1)
        
        n_chunks = len(self._df_list)
        testfrac = 1.0 - self._trainval_frac
        # randomiza que chunks van a train o test siguiendo la fraccion testfrac
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)
 
        # Recorre todos los bloques, separa en train/set y convierte el output a categorical
        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_by_std, only_out_of_range)
            if bad_block:
                print("bad block")
                continue
            if lag_blocks_x.size == 0:
                print("zero block")
                continue
            self._boost_data += boost_blocks.ravel().tolist()

            # conversion de boost a clases categoricas
            rngs = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)

            # asignacion del bloque a train/test
            if flag == 'test':
                test_x1 = numpy.vstack([test_x1, lag_blocks_x])
                test_y1 = numpy.vstack([test_y1, rngs])
                self._testing_keys.append(df_key)
                nsamp_test += len(df)
            if flag == 'train':
                train_x1 = numpy.vstack([train_x1, lag_blocks_x])
                train_y1 = numpy.vstack([train_y1, rngs])
                self._training_keys.append(df_key)
                nsamp_train += len(df)
            print(df_key," testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::],]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::],]


