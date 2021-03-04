import pandas
import numpy

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
    
    _has_star_change_fields = False
    _df_ra_field = None
    _df_dec_field = None
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
    
    def set_ra_dec_field_name(self, ra_field_name, dec_field_name):
        '''
        '''
        self._df_ra_field = ra_field_name
        self._df_dec_field = dec_field_name
        self._has_star_change_fields = True
          
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
    
    def build_star_change_flag_fields(self, dataframe):
        '''
        '''
        if not self._has_star_change_fields:
            raise RuntimeError("RA/dec fields are not specified")
        ra = dataframe[self._df_ra_field]
        dec = dataframe[self._df_dec_field]
        diff_ra = numpy.abs(ra - ra.shift(-1))
        diff_dec = numpy.abs(dec - dec.shift(-1))
        diff_ra[diff_ra > 0] = 1
        diff_dec[diff_dec > 0] = 1
        # discard last item because it is a NaN
        diff_ra = numpy.concatenate([[0,], diff_ra.values[0:-1]])
        diff_dec = numpy.concatenate([[0,], diff_dec.values[0:-1]])
        diff_ra = diff_ra.astype('int32')
        diff_dec = diff_dec.astype('int32')
        dataframe['diff_ra'] = diff_ra
        dataframe['diff_dec'] = diff_dec
        return dataframe

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

    def blocks_from_dataframe(self, 
        dataframe, 
        filter_star_change, only_star_change, 
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
        # auxiliary variables for star change filtering
        diff_ra_data = None
        diff_dec_data = None
        ##print("filter_star_change= ",filter_star_change)
        if filter_star_change:
            dataframe = self.build_star_change_flag_fields(dataframe)
            diff_ra_data = dataframe['diff_ra'].values
            diff_dec_data = dataframe['diff_dec'].values
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
            out_of_range = False
            if filter_by_std:
                out_of_range = self.check_block_variance(boost_data)
                if only_out_of_range:
                    out_of_range = not out_of_range
            if out_of_range:
                ix += self._block_stride
                iy += self._block_stride
                continue
            # filter star changes
            # deleted this star change block ROB

            lag_blocks_x.append(lag_data_x)
            lag_blocks_y.append(lag_data_y)
            boost_blocks.append(boost_data)
            
            ix += self._block_stride
            iy += self._block_stride
        
        lag_blocks_x = numpy.asarray(lag_blocks_x, dtype='float32')
        lag_blocks_y = numpy.asarray(lag_blocks_y, dtype='float32')
        boost_blocks = numpy.asarray(boost_blocks, dtype='float32')
        
        return bad_block, lag_blocks_x, lag_blocks_y, boost_blocks

class MaxContinuous(BaseHandler):
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

    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        y_data = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            # find min/max of average of n consecutive points in boost
            ii = 0
            max_avg = -1e10
            while ii + self.n_max_avg < len(boost_data):
                yw = boost_data[ii:ii+self.n_max_avg, :].ravel()
                local_avg = numpy.average(yw)
                if local_avg > max_avg:
                    max_avg = local_avg
                ii += 1
            y_data.append([max_avg,])
        y_data = numpy.asarray(y_data, dtype='float32')        
        return y_data

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
        filter_by_std=False, only_out_of_range=False):
        '''
        '''
        if self._trainval_frac < 0 or self._trainval_frac > 1.0:
            raise ValueError("training/val fraction must be in the ]0,1[ range")

        shape_x1 = (1, self._lag, len(self._x_col_names))
        shape_y1 = (1, 1)
        
        test_x1 = numpy.zeros(shape_x1)
        test_y1 = numpy.zeros(shape_y1)
        
        train_x1 = numpy.zeros(shape_x1)
        train_y1 = numpy.zeros(shape_y1)
        
        n_chunks = len(self._df_list)
        testfrac = 1.0 - self._trainval_frac
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)

        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                continue
            if lag_blocks_x.size == 0:
                continue
            maxes = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)
            if flag == 'test':
                test_x1 = numpy.vstack([test_x1, lag_blocks_x])
                test_y1 = numpy.vstack([test_y1, maxes])
                self._testing_keys.append(df_key)
                nsamp_test += len(df)
            if flag == 'train':
                train_x1 = numpy.vstack([train_x1, lag_blocks_x])
                train_y1 = numpy.vstack([train_y1, maxes])
                self._training_keys.append(df_key)
                nsamp_train += len(df)
            print("testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::],]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::],]

class MinContinuous(BaseHandler):
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

    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        y_data = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            # find min/max of average of n consecutive points in boost
            ii = 0
            min_avg = 1e10
            while ii + self.n_max_avg < len(boost_data):
                yw = boost_data[ii:ii+self.n_max_avg, :].ravel()
                local_avg = numpy.average(yw)
                if local_avg < min_avg:
                    min_avg = local_avg
                ii += 1
            y_data.append([min_avg,])
        y_data = numpy.asarray(y_data, dtype='float32')        
        return y_data

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
        filter_by_std=False, only_out_of_range=False):
        '''
        '''
        if self._trainval_frac < 0 or self._trainval_frac > 1.0:
            raise ValueError("training/val fraction must be in the ]0,1[ range")

        shape_x1 = (1, self._lag, len(self._x_col_names))
        shape_y1 = (1, 1)
        
        test_x1 = numpy.zeros(shape_x1)
        test_y1 = numpy.zeros(shape_y1)
        
        train_x1 = numpy.zeros(shape_x1)
        train_y1 = numpy.zeros(shape_y1)
        
        n_chunks = len(self._df_list)
        testfrac = 1.0 - self._trainval_frac
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)

        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                continue
            if lag_blocks_x.size == 0:
                continue
            mins = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)
            if flag == 'test':
                test_x1 = numpy.vstack([test_x1, lag_blocks_x])
                test_y1 = numpy.vstack([test_y1, mins])
                self._testing_keys.append(df_key)
                nsamp_test += len(df)
            if flag == 'train':
                train_x1 = numpy.vstack([train_x1, lag_blocks_x])
                train_y1 = numpy.vstack([train_y1, mins])
                self._training_keys.append(df_key)
                nsamp_train += len(df)
            print("testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::],]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::],]

class AvgContinuous(BaseHandler):
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

    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        y_data = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            yw = boost_data.ravel()
            local_avg = numpy.average(yw)
            y_data.append([local_avg,])
        y_data = numpy.asarray(y_data, dtype='float32')        
        return y_data

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
        filter_by_std=False, only_out_of_range=False):
        '''
        '''
        if self._trainval_frac < 0 or self._trainval_frac > 1.0:
            raise ValueError("training/val fraction must be in the ]0,1[ range")

        shape_x1 = (1, self._lag, len(self._x_col_names))
        shape_y1 = (1, 1)
        
        test_x1 = numpy.zeros(shape_x1)
        test_y1 = numpy.zeros(shape_y1)
        
        train_x1 = numpy.zeros(shape_x1)
        train_y1 = numpy.zeros(shape_y1)
        
        n_chunks = len(self._df_list)
        testfrac = 1.0 - self._trainval_frac
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)

        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                continue
            if lag_blocks_x.size == 0:
                continue
            avgs = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)
            if flag == 'test':
                test_x1 = numpy.vstack([test_x1, lag_blocks_x])
                test_y1 = numpy.vstack([test_y1, avgs])
                self._testing_keys.append(df_key)
                nsamp_test += len(df)
            if flag == 'train':
                train_x1 = numpy.vstack([train_x1, lag_blocks_x])
                train_y1 = numpy.vstack([train_y1, avgs])
                self._training_keys.append(df_key)
                nsamp_train += len(df)
            print("testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::],]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::],]

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

    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        rngs = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            # find min/max of average of n consecutive points in boost
            #ii = 0
 
            #ROB max de utlimas 8 horsas
            ii = len(boost_data)-10
            if ii<0:
                ii=0

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

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
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
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)
        
        #print(test_or_train)

        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                print("bad block")
                continue
            if lag_blocks_x.size == 0:
                print("zero block")
                continue
            self._boost_data += boost_blocks.ravel().tolist()
            rngs = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)
            #print('df=',df.shape,'shape lag_blocks_x',lag_blocks_x.shape)
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

class MinCategorical(BaseHandler):
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

    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        rngs = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            # find min/max of average of n consecutive points in boost
            ii = 0
            min_avg = 1e10
            while ii + self.n_max_avg < len(boost_data):
                yw = boost_data[ii:ii+self.n_max_avg, :].ravel()
                local_avg = numpy.average(yw)
                if local_avg < min_avg:
                    min_avg = local_avg
                ii += 1
            # truncate negative values
            if min_avg < 0:
                min_avg = 1e-5
            ranges = numpy.zeros(self.n_ranges, dtype='float32')                
            for i_r in range(self.n_ranges):
                lr = self.ranges[i_r]
                rr = self.ranges[i_r+1]
                if min_avg >= lr and min_avg < rr:
                    ranges[i_r] = 1.0
            ranges_ok = numpy.sum(ranges) == 1.0
            # this should never happen
            if not ranges_ok:
                raise RuntimeError("bad range output")
            rngs.append(ranges)
        rngs = numpy.asarray(rngs, dtype='float32')        
        return rngs

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
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
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)

        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                continue
            if lag_blocks_x.size == 0:
                continue
            rngs = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)
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
            print("testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::],]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::],]

class AvgCategorical(BaseHandler):
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

    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        rngs = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            # find min/max of average of n consecutive points in boost
            ii = 0
            yw = boost_data.ravel()
            local_avg = numpy.average(yw)
            # truncate negative values
            if local_avg < 0:
                local_avg = 1e-5
            ranges = numpy.zeros(self.n_ranges, dtype='float32')                
            for i_r in range(self.n_ranges):
                lr = self.ranges[i_r]
                rr = self.ranges[i_r+1]
                if local_avg >= lr and local_avg < rr:
                    ranges[i_r] = 1.0
            ranges_ok = numpy.sum(ranges) == 1.0
            # this should never happen
            if not ranges_ok:
                raise RuntimeError("bad range output")
            rngs.append(ranges)
        rngs = numpy.asarray(rngs, dtype='float32')        
        return rngs

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
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
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)

        nsamp_test = 0
        nsamp_train = 0
        boost_data = list()
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                continue
            if lag_blocks_x.size == 0:
                continue
            boost_data += boost_blocks.ravel().tolist()
            rngs = self.blocks_to_dnn_input(lag_blocks_x, lag_blocks_y, boost_blocks)
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
            print("testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::],]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::],]
        
        self._boost_data = boost_data

class TrendCategorical(BaseHandler):
    '''
    '''
    def __init__(self, path_to_hdf, x_col_names, name_of_y_col, max_df=-1):
        '''
        '''
        super().__init__(path_to_hdf, x_col_names, name_of_y_col, max_df)
                
        self.n_ranges_up = 0
        self.ranges_up = list()
        self.n_ranges_down = 0
        self.ranges_down = list()
        self.n_past_avg = 0
        self.n_max_avg = 0
                
    def set_up_trend_ranges(self, up_ranges):
        '''
        '''
        self.n_ranges_up = len(up_ranges)
        self.ranges_up = up_ranges + [1e10,]

    def set_down_trend_ranges(self, down_ranges):
        '''
        '''
        self.n_ranges_down = len(down_ranges)
        self.ranges_down = down_ranges + [1e10,]

    def set_boost_avg_steps(self, n):
        '''
        '''
        self.n_max_avg = n

    def set_lag_avg_steps(self, n):
        '''
        '''
        self.n_past_avg = n
    
    def blocks_to_dnn_input(self, lag_blocks_x, lag_blocks_y, boost_blocks):
        '''
        '''
        rngs_up = list()
        rngs_down = list()
        zf = zip(lag_blocks_x, lag_blocks_y, boost_blocks)
        for lag_data_x, lag_data_y, boost_data in zf:
            # find min/max of average of n consecutive points in boost
            ii = 0
            max_avg = -1e10
            min_avg = 1e10
            while ii + self.n_max_avg < len(boost_data):
                yw = boost_data[ii:ii+self.n_max_avg, :].ravel()
                local_avg = numpy.average(yw)
                if local_avg > max_avg:
                    max_avg = local_avg
                if local_avg < min_avg:
                    min_avg = local_avg
                ii += 1
            # compute average of past n_steps
            past_avg = numpy.average(lag_data_y[-self.n_past_avg: ,])
            max_delta = max_avg - past_avg
            min_delta = past_avg - min_avg
            # truncate negative values
            if max_delta < 0:
                max_delta = 1e-5
            if min_delta < 0:
                min_delta = 1e-5
            # assemble categorical output for up-trend
            rel_ranges_up = numpy.zeros(self.n_ranges_up, dtype='float32')                
            for i_r in range(self.n_ranges_up):
                lr = self.ranges_up[i_r]
                rr = self.ranges_up[i_r+1]
                if max_delta >= lr and max_delta < rr:
                    rel_ranges_up[i_r] = 1.0
            # assemble categorical output for down-trend
            rel_ranges_down = numpy.zeros(self.n_ranges_down, dtype='float32')
            for i_r in range(self.n_ranges_down):
                lr = self.ranges_down[i_r]
                rr = self.ranges_down[i_r+1]
                if min_delta >= lr and min_delta < rr:
                    rel_ranges_down[i_r] = 1.0      
            # check ranges have data
            range_up_ok = numpy.sum(rel_ranges_up) == 1.0
            range_dw_ok = numpy.sum(rel_ranges_down) == 1.0
            # this should never happen
            if not range_up_ok:
                print(min_delta, max_delta, numpy.sum(rel_ranges_up))
                raise RuntimeError("bad up output")
            # this should never happen
            if not range_dw_ok:
                raise RuntimeError("bad down output")
            rngs_up.append(rel_ranges_up)
            rngs_down.append(rel_ranges_down)
            
        rngs_up = numpy.asarray(rngs_up, dtype='float32')
        rngs_down = numpy.asarray(rngs_down, dtype='float32')
        
        return rngs_up, rngs_down

    def build_dataset(self,
        filter_star_change=False, only_star_change=False, 
        filter_by_std=False, only_out_of_range=False):
        '''
        '''
        if self._trainval_frac < 0 or self._trainval_frac > 1.0:
            raise ValueError("training/val fraction must be in the ]0,1[ range")

        shape_x1 = (1, self._lag, len(self._x_col_names))
        shape_y1 = (1, self.n_ranges_up)
        shape_y2 = (1, self.n_ranges_down)
        
        test_x1 = numpy.zeros(shape_x1)
        test_y1 = numpy.zeros(shape_y1)
        test_y2 = numpy.zeros(shape_y2)
        
        train_x1 = numpy.zeros(shape_x1)
        train_y1 = numpy.zeros(shape_y1)
        train_y2 = numpy.zeros(shape_y2)
        
        n_chunks = len(self._df_list)
        testfrac = 1.0 - self._trainval_frac
        test_or_train = numpy.random.choice(
            ['test', 'train'], p=(testfrac, self._trainval_frac), 
            size=n_chunks, replace=True)

        nsamp_test = 0
        nsamp_train = 0
        self._hdf_store.open()
        for df_key, flag in zip(self._df_list, test_or_train):
            df = pandas.read_hdf(self._hdf_store, df_key)
            bad_block, lag_blocks_x, lag_blocks_y, boost_blocks = \
                self.blocks_from_dataframe(
                    df, 
                    filter_star_change, only_star_change,
                    filter_by_std, only_out_of_range)
            if bad_block:
                continue
            if lag_blocks_x.size == 0:
                continue
            rngs_up, rngs_down = self.blocks_to_dnn_input(
                lag_blocks_x, lag_blocks_y, boost_blocks)
            if flag == 'test':
                test_x1 = numpy.vstack([test_x1, lag_blocks_x])
                test_y1 = numpy.vstack([test_y1, rngs_up])
                test_y2 = numpy.vstack([test_y2, rngs_down])
                self._testing_keys.append(df_key)
                nsamp_test += len(df)
            if flag == 'train':
                train_x1 = numpy.vstack([train_x1, lag_blocks_x])
                train_y1 = numpy.vstack([train_y1, rngs_up])
                train_y2 = numpy.vstack([train_y2, rngs_down])
                self._training_keys.append(df_key)
                nsamp_train += len(df)
            print("testing nsamp = ", nsamp_test, "training nsamp = ", nsamp_train)
        self._hdf_store.close()
        
        self._train_x += [train_x1[1::],]
        self._train_y += [train_y1[1::], train_y2[1::]]
        
        self._test_x = [test_x1[1::],]
        self._test_y = [test_y1[1::], test_y2[1::]]
