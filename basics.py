import os
import pathlib

import json
import radiomics
import radiomics.imageoperations as iop
import SimpleITK as sitk
import sqlite3

from cenotaph.basics.generic_functions import get_folders

class Calculator():
    """Calculator of radiomics features"""
    
    #List of features that are computed via pyradiomics directly
    _pyradiomics_lut = {'Maximum' : ['firstorder', 'Maximum'],
                        'Minimum' : ['firstorder', 'Minimum'],
                        'Mean' : ['firstorder', 'Mean'],
                        'StdDev' : ['firstorder', 'StandardDeviation'],
                        'Skewness' : ['firstorder', 'Skewness'],
                        'Kurtosis' : ['firstorder', 'Kurtosis'],
                        'Uniformity' : ['firstorder', 'Uniformity'],
                        'GLCM_Contrast' : ['glcm', 'Contrast'],
                        'GLCM_Correlation' : ['glcm', 'Correlation'],
                        'GLCM_DiffVar' : ['glcm', 'DifferenceVariance'],
                        'GLCM_Energy' : ['glcm', 'JointEnergy'],
                        'GLCM_Entropy' : ['glcm', 'JointEntropy'],
                        'NGTDM_Busyness' : ['ngtdm', 'Busyness'],
                        'NGTDM_Coarseness' : ['ngtdm', 'Coarseness'],
                        'NGTDM_Complexity' : ['ngtdm', 'Complexity'],
                        'NGTDM_Contrast' : ['ngtdm', 'Contrast'],
                        'Elongation' : ['shape', 'Elongation'],
                        'Flatness' : ['shape', 'Flatness'],
                        'MaxAxialDiameter' : ['shape', 
                                              'Maximum2DDiameterSlice'],
                        'Sphericity' : ['shape', 'Sphericity'],
                        'MeshVolume' : ['shape', 'MeshVolume'],
                        'VoxelVolume' : ['shape', 'VoxelVolume']
                       }
                   
    def __init__(self, bin_count=256, cache='./cache.db', 
                 features_to_compute=[], first_level_subdir='CT', 
                 image_name='image.nrrd', mask_name='mask.nii', root='./root',
                 **kwargs):
        """ The constructor
            
        Parameters
        ----------
        bin_count
            The number of histogram bins for histogram-based
            calculations.                           
        cache
            Path to the database (.db) file where the results are to be cached.
        features_to_compute
            An array of [class, feature, windowing] triplets indicating the
            features to compute. 
                class     -> see Pyradiomics documentation for possible
                             values.
                feature   -> see Pyradiomics documentation for possible
                                 values.
                windowing -> A flag indicating whether to apply windowing
                            should when computing this feature. Possible
                            values are:
                                'windowed'
                                'not-windowed'
        first_level_subdir : str
            Defines the first-level directory (below 'root') where the data are
            stored. This usually represents the imaging modality (i.e. CT, PET
            or MRI).
        image_name : str
            The name of the file representing the source scan, must be in 
            first_level_subdir. Currently supported format is SimpleITK nrrd.
        mask_name : str
            The name of the file representing the lesion segmenation (region
            of interest). Must be in first_level_subdir. Currently supported 
            format is SimpleITK nii.
        root : str
            Path to folder where the input data are stored. The directory
            tree needs to be organised as follows:
                root/case-001/first-level-subdir/image-file, mask_file
                root/case-002/first-level-subdir/image-file, mask_file
                           ...
        window : float, array-like (2)
            An array of two values indicating the lower and upper limits 
            (accepted range or window). Voxels with values outside the limits
            are excluded from the analysis. If the parameter is given,
            histogram-based features are computed through absolute scaling
            (the histogram bounds are the window bounds). Otherwise the lower
            and upper bound of the histogram are the minimum and maximum value
            in the ROI (relative scaling). The parameter is required if any of
            the features to compute is 'windowed'
        """
    
        self._bin_count = bin_count
        self._cache = cache
        self._first_level_subdir = first_level_subdir
        self._image_name = image_name
        self._mask_name = mask_name
        self._root = root
            
        #Parse the optional settings
        opt_params = kwargs.keys()
        if 'window' in opt_params:
            self._window = kwargs['window']
        if 'features_to_compute' in opt_params:
            self._features_to_compute = config_data['features_to_compute']
            
        #Set up the connection to the database
        self._db = DBDriver(self._cache)  

    @classmethod
    def generate_calculator_from_file(cls, config_file):
        """Generate a radiomic calculator from a set of configuration options
        stored in a .json file
    
        Parameters
        ----------
        config_file : str
            Pointer to the json file where the configuration parameters are
            stored. For the tags admitted see generate_calculator_from_dict()
        Returns
        -------
        calculator : RadiomicCalculator
            The radiomic calculator
        """
            
        #Read the settings from the configuration file
        with open(config_file) as fp:
            config_data = json.load(fp)
        
    
        return cls.generate_calculator_from_dict(config_data)

    @staticmethod 
    def generate_calculator_from_dict(options):
        """Generate a radiomic calculator from a set of configuration options
        passed as a dictionary
            
        Parameters
        ----------
        options : dict
            A dictionary containing the configuration parameters. The tags 
            admitted are defined below.
                BinCount (int)
                    -> The number of histogram bins for histogram-based
                       calculations.                           
                Cache (str)
                    -> Path to the database (.db) file where the results
                       are to be cached.
                FeaturesToCompute (array of [str,str])
                    -> An array of [feature_name, windowing] pairs
                       indicating the features to compute. 
                            feature_name -> The name of the feature to compute.
                                            See get_feature_value() for
                                            possible values.
                            windowing    -> Fag indicating whether to apply
                                            windowing when computing (also see 
                                            'Windowing' and 'BinWidth') 
                                            parameters. Possible values
                                            are: 'windowed' and 'not-windowed'
                FirstLevelSubdir (str)
                    -> First-level directory under the case folder. Usually 
                       represents the imaging modality (i.e. 'CT', 'PET' or 
                       'MRI').
                ImageName (str)
                    -> The name of the file representing the source scan, must
                       be in FirstLevelSubdir. Currently supported format is 
                       SimpleITK nrrd.
                MaskName (str)
                    -> The name of the file representing the lesion segmenation
                       (region of interest). Must be in FirstLevelSubdir. 
                       Currently supported format is SimpleITK nii.
                Root (str)
                        -> Base folder where the input data are stored. 
                           The directory tree needs to be organised as follows:
                               root/
                                   case-001/
                                           first-level-subdir/
                                                              image-file, 
                                                              mask_file
                               root/
                                   case-002/
                                           first-level-subdir/
                                                              image-file, 
                                                              mask_file
                               ...
                Window (optional)
                    -> An array of two values indicating the lower and upper 
                       limits (accepted range or window). Voxels with values 
                       outside the limits are excluded from the analysis.
                       If the parameter is given, histogram-based features are
                       computed through absolute scaling (the histogram bounds
                       are the window bounds). Otherwise the lower and upper 
                       bound of the histogram are the minimum and maximum
                       value in the ROI (relative scaling).
            
        Returns
        -------
        calculator : Calculator
            The radiomic calculator
        """
        #Parse the required settings, return an exception if some are missing
        try:
            bin_count = options['BinCount']
            cache = options['Cache']
            features_to_compute = options['FeaturesToCompute']
            first_level_subdir = options['FirstLevelSubdir']  
            root = options['Root']
        except KeyError as kerr:
            raise Exception('Parameter {} not provided in the configuration '
                            'file'.format(kerr))
            
        #Create and return the calculator
        calculator = Calculator(bin_count = bin_count, 
                                cache = cache, 
                                features_to_compute = features_to_compute, 
                                first_level_subdir = first_level_subdir,
                                root = root)
            
        #Parse the optional settings and update the calculator
        settings = options.keys()
        if 'Windowing' in settings:
            calculator.set_window(options['Windowing'])
        if 'FeaturesToCompute' in settings:
            calculator.set_features_to_compute(options['FeaturesToCompute']) 
                
        return calculator
        
    def get_feature_value(self, case_id, feature_name, num_bins = 256, 
                          windowed = 'not-windowed'):
        """Get the feature value for a given case
        
        Parameters
        ----------
        case_id : str
            The case ID
        feature_name : str
            The name of the feature to compute. Possible values are:
                'Mean'             -> Average value
                'Maximum'          -> Maximum value
                'Minimum'          -> Minimum value
                'StdDev'           -> Standard deviation
                'Skewness'         -> Skewness
                'Kurtosis'         -> Kurtosis
                'Uniformity'       -> Uniformity
                'GLCM_Contrast'    -> GLCM contrast
                'GLCM_Correlation' -> GLCM correlation
                'GLCM_DiffVar'     -> GLCM difference variance
                'GLCM_Energy'      -> GLCM energy
                'GLCM_Entropy'     -> GLCM entropy
                'Volume'           -> Volume (via mesh calculation)
                'MaxAxialDiameter' -> Maximum diameter on axial slices
                'Elongation'       -> Elongation
                'Flatness'         -> Flatness
                'Sphericity'       -> Sphericity  
        num_bins : int
            Number of bins used for computing histogram-based features
        windowed : str
            Flag indicating whether to apply windowing or not. Can be:
                "windowed" (windowed) or "not-windowed" (not windowed)
            
        Returns
        -------
        value : float
            The feature value
        """
        
        #Generate string for database lookup
        window_str = '_w'
        window_flag = True
        if windowed == "not-windowed":
            window_str = '_nw'
            window_flag = False
        feature_name_db = feature_name + window_str
        
        
        #Check if the feature is stored in the cache; if not, compute and store
        #it in the cache
        feature_exists = self._db.feature_exists(feature_name_db)
        if feature_exists:
            feature_not_null = self._db.get_feature_value(case_id,
                                                          feature_name_db)
        if feature_exists and feature_not_null:
            print('Retrieving {} ({}) on case {}'.format(
                feature_name, windowed, case_id), end = '')
            value = feature_not_null
        else:
            print('Computing {} ({}) on case {}'.format(
                feature_name, windowed, case_id), end = '')
            value = self._compute_feature_value(case_id, 
                                                feature_name, 
                                                num_bins,
                                                window_flag)
            self._db.set_feature_value(case_id,  feature_name_db, value,
                                       data_type = 'REAL')
        print(' ({:.2f})'.format(value))
        
        return value
    
    def _compute_feature_value(self, case_id, feature_name, num_bins,
                               windowed):
        """Compute the feature value for a given case
        
        Parameters
        ----------
        see get_feature_value()
        
        Returns
        -------
        value : float
            The feature value
        """
        
        #Get the image and the mask
        image_source = pathlib.PurePath(self._root, 
                                        case_id,
                                        self._first_level_subdir,
                                        self._image_name)
        mask_source = pathlib.PurePath(self._root, 
                                       case_id,
                                       self._first_level_subdir,
                                       self._mask_name)        
        
        try:
            image = sitk.ReadImage(str(image_source))
            mask = sitk.ReadImage(str(mask_source))
        except FileNotFoundError:
            raise Exception('Error opening image and/or mask')        
        
        #Pass the window if required
        if windowed:
            opt_args = {'window' : self._window}
        else:
            opt_args = {}
            
        if feature_name in self._pyradiomics_lut.keys():
            #If the feature is in the group computed via pyradiomics dispatch 
            #the call to the pyradiomics wrapper         
            value = PyradiomicsWrapper.compute_feature_one_case(
                    image, 
                    mask, 
                    self._pyradiomics_lut[feature_name][0],
                    self._pyradiomics_lut[feature_name][1],
                    num_bins,
                    **opt_args)
        else:
            raise Exception('Feature {} not implemented'.format(feature_name))   
        
        return value.item()
    
    def set_window(self, window):
        """Set the window for feature calculation
        
        Parameters
        ----------
        window : (float, float)
            The lower and upper bound of the window
        """
        
        if not len(window) == 2:
            raise Exception('The window needs to have two values')
        if not (window[0] < window[1]):
            raise Exception('The lower window bound needs to be strictly'
                            'less than the upper bound')
        
        self._window = window
        
    def set_features_to_compute(self, features_to_compute):
        """Store the features to be computed
        
        Parameters
        ----------
        features_to_compute : list (4)
            The features to compute. A list of four parameters indicating:
                1) the feature name (str)
                2) whether to apply windowing ('windowed') or not 
                   ('not-windowed'), (str)
                3) the unit of measure for that feature (str)
                4) the number of decimal places to be used when displaying the
                   feature value
        """
        
        windowing_flags = ['windowed', 'not-windowed']
        if not len(features_to_compute) > 0:
            raise Exception('There are no features to compute')
        
        for f in features_to_compute:
            if not len(f) == 4:
                raise Exception('Features to compute need to be passed as'
                                '(name, windowing flag, unit) triples')
            if not f[1] in windowing_flags:
                raise Exception('Windowing flag not recognised')
            
        self._features_to_compute = features_to_compute
                        
    def get_cases(self):
        """List of cases contained in the root folder
        
        Returns
        -------
        cases : list of str
            The list containing the case names
        """
        return get_folders(self._root)       
        

class PyradiomicsWrapper():
    """Wrapper around Pyradiomics"""
    
    #********************************************
    #************** Class variables *************
    #********************************************
    #Lookup table for classes of radiomic features
    feature_classes_lut = {'glcm' : radiomics.glcm.RadiomicsGLCM,
                           'gldm' : radiomics.gldm.RadiomicsGLDM,
                           'glszm' : radiomics.glszm.RadiomicsGLSZM,
                           'glrlm' : radiomics.glrlm.RadiomicsGLRLM,
                           'firstorder' : radiomics.firstorder.RadiomicsFirstOrder,
                           'ngtdm' : radiomics.ngtdm.RadiomicsNGTDM,
                           'shape': radiomics.shape.RadiomicsShape}
    #********************************************
    #********************************************
    #********************************************
    
        
    @classmethod         
    def compute_feature_one_case(cls, image, mask, feature_class, feature_name, 
                                 num_bins, **kwargs):
        """Compute the given feature for one single case
            
        Parameters
        ----------
        image : SimpleITK.Image
            The input image
        mask : SimpleITK.Image
            The mask defining the region of interest 
        feature_class : str
            The feature class.
        feature_name : str
            The feature name
        num_bins : int
            Number of bins used for computing histogram-based features
        window (optional) : list of floats (2)
            The lower and upper bound of the window (if windowing is required)
                
        Returns
        -------
        value : float
            The feature value
        """
            
        #Raise an exception if the feature class is unknown 
        if feature_class not in cls.feature_classes_lut.keys():
            raise Exception("Feature class unknown")
            
        #Set up the feature calculator for this feature class
        #If no window is not given perform relative scaling (between min
        #and max of the ROI)
        if not 'window' in kwargs.keys():
            calculator = cls.\
                feature_classes_lut[feature_class](image, mask,
                                                   binCount = num_bins)
            
        #If a windowed is given perform absolute scaling (between min and
        #max of the window)
        else:
            window_width = (kwargs['window'][1] - kwargs['window'][0])
            bin_width = window_width/num_bins
            calculator = cls.\
                feature_classes_lut[feature_class](image, mask,
                                                   binWidth = bin_width)
            
        #Enable calculation of the features required
        calculator.enableFeatureByName(feature_name)
                
        #Compute the features
        calculator.execute()
                
        return calculator.featureValues[feature_name]    
                
    def _compute_feature_value(self, case_id, class_name, feature_name,
                               windowed):
        """Compute the feature value for a given case
        
        Parameters
        ----------
        case_id : str
            The case ID
        class_name : str
            The name of the feature class. Possible values are:
                'glcm', 'gldm', 'glszm', 'glrlm', 'firstorder', 'ngtdm'
                and 'shape'. See also Pyradiomics documentation for updates
        feature_name : str
            The feature name, which depends on the feature class. For possible
            values see Pyradiomcs documentation.
        windowed : bool
            Flag indicating whether to apply windowing or not
            
        Returns
        -------
        value : float
            The feature value
        """  
        

        
        #Apply windowing if required
        if windowed:    
            try:
                mask = iop.resegmentMask(image, mask,
                                         resegmentRange = self._window)
            except AttributeError:
                raise Exception('Window not defined')
        
        #Compute and return the feature        
        value = self._compute_features_one_case(image,
                                                mask,
                                                windowed,
                                                [(class_name, feature_name)])
        value = value[0].item()
        return value        
            
class DBDriver():
    """Database interface"""
    
    def __init__(self, dbfile, table_name = 'Data'):
        """The constructor
        
        Parameters
        ----------
        dbfile : str
            Path to the database file
        table_name : str
            The name of the table where the data will be stored
        """
        
        #Establish a connection
        try:
            self._connection = sqlite3.connect(dbfile)
        except:
            raise Exception('Cannot establish a connection to {}'
                            .format(dbfile))
        self._cursor = self._connection.cursor()
        
        #Create the data table if it doesn't exist
        self._table_name = table_name
        self._case_id_col = 'CaseID'
        self._cursor.execute("CREATE TABLE if not exists "
                             + self._table_name 
                             + " ({})".format(self._case_id_col))
        self._commit()
        
    def _commit(self):
        """Commit a change"""
        self._connection.commit()
        
    def get_feature_value(self, case_id, feature_name):
        """Get the value for a given feature and case (table cell). 
        
        Parameters
        ----------
        case_id : str
            The case ID
        feature_name : str
            The feature name
            
        Returns
        -------
        value : ?
            The feature value
        """   
        
        sql_command = "SELECT " + feature_name + " FROM "\
                      + "'" + self._table_name + "'" + " WHERE "\
                      + self._case_id_col + "=" + "'" + case_id + "'"
        sql_result = self._cursor.execute(sql_command).fetchone()
        if not sql_result == None:            
            value = sql_result[0]
        else:
            value = None
        return value
        
      
        
    def set_feature_value(self, case_id, feature_name, value, data_type):
        """Set the value for a given feature and case (table cell). Any
        previous value will be ovewritten.
        
        Parameters
        ----------
        case_id : str
            The case ID
        feature_name : str
            The feature name
        value : depends on data_type
            The value to be inserted
        data_type : str
            The feature datatype. See sqlite documentation for possible values
        """
        
        #Check if the given feature and case are already in the database
        if not self.feature_exists(feature_name):
            self.add_feature(feature_name, data_type = 'REAL')
        if not self.case_exists(case_id):
            self.add_case(case_id)
            
        sql_command = "UPDATE " + self._table_name + " SET "\
                      + "'" + feature_name + "'" + "=" + value.__str__()\
                      + " WHERE " + self._case_id_col + "="\
                      + "'" + case_id + "'"
        self._cursor.execute(sql_command)
        self._commit()        
        
    def add_case(self, case_id):
        """Add a new case as an empty record. If the record is already present
        no new record is created.
        
        Parameters
        ----------
        case_id : str
            The case ID
        """
        if not self.case_exists(case_id):
            sql_command = "INSERT INTO " + self._table_name\
                          + " ({})".format(self._case_id_col)\
                          + " VALUES ('{}')".format(case_id)
            self._cursor.execute(sql_command)
            self._commit()
        
    def case_exists(self, caseID):
        """Check if the record corresponding to the given file has been 
        already added to the table
        
        Parameters
        ----------
        caseID : str
            The case ID

        Returns
        -------
        exists : bool
            Flag indicating whether the record exists (True) or not (False) 
        """
        exists = False
        sql_command = "SELECT * FROM " + self._table_name + " WHERE "\
                      + self._case_id_col + "=" + "'" + caseID + "'"
        sql_result = self._cursor.execute(sql_command)
        if len(sql_result.fetchall()) > 0:
            exists = True
        
        return exists
    
    def add_feature(self, feature_name, data_type = 'NULL'):
        """Add a new feature placeholder as an empty column. If the feature 
        is already present no new column is created.
        
        Parameters
        ----------
        feature_name : str
            The feature name
        data_type : str
            The feature datatype. See sqlite documentation for possible values
        """
        if not self.feature_exists(feature_name):
            sql_command = "ALTER TABLE " + self._table_name + " ADD "\
                          + feature_name + " " + data_type
            sql_result = self._cursor.execute(sql_command)
            self._commit()
    
    def feature_exists(self, feature_name):
        """Check if the given feature name exists as a column in the database
        
        Parameters
        ----------
        feature_name : str
            The case ID

        Returns
        -------
        exists : bool
            Flag indicating whether the record exists (True) or not (False) 
        """
        exists = False
        sql_command = "SELECT COUNT(*) AS CNTREC FROM pragma_table_info("\
                      + "'" + self._table_name + "'" + ") WHERE name="\
                      + "'" + feature_name + "'"
        
        sql_result = self._cursor.execute(sql_command)
        if sql_result.fetchone()[0] > 0:
            exists = True
       
        return exists    

    
            
            
