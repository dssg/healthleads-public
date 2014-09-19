class DAL:
    # Data Access Layer
    def __init__(self, path_in, path_out):
        self.root_in = path_in
        self.root_out = path_out
    def save(self, out_name, obj):
        '''
        Input: out_name : string
               obj : string-like (json, csv, pickle)
        output: errors that we encountered when writing the file
        '''
        
        f = open(self.root_out + out_name, "w")
        e = f.write(obj)
        f.close()
        return e
    def load(self, in_name):
        '''
        Input: in_name: string
        Output: f is the file that was opened 
        '''
        f = open(self.root_in + in_name,"r")
        return f 
    
    def csv_look(self): 
        '''
        Input: none
        Output: directory listing for input and output directories
        '''
        import glob as g
        in_files = g.glob(self.root_in + "*.csv")
        out_files = g.glob(self.root_out + "*.csv")
        return (in_files,out_files)
    
    def load_csv(self,csvin,opts, clean = True, infer_dt_form = False):
        '''
        Input:  csvin : string ( name of csv without extention ),
                opts : dict ( the na_value settings) 
        Output: the pandas dataframe
        '''
        import pandas as pd
        if not clean:
            return pd.read_csv(self.root_in + csvin + ".csv", na_values=opts,low_memory=False,infer_datetime_format=infer_dt_form)
        else:
            return pd.read_csv(self.root_out + csvin + ".csv", na_values=opts,low_memory=False,infer_datetime_format=infer_dt_form)
            
            
    
    def save_csv(self,csvout,df,opts):
        '''
        Input:    csvout : string ( name of csv without extention ),
                  df : dataframe ( pandas dataframe that we want to save)
                 opts : string ( how we will represent the nan) 
        Output: the errors that occur when saving the file
        '''
        e = df.to_csv(self.root_out + csvout + ".csv", na_rep = opts, index=False )
        return e
    def remove_csv(self,csvname):
        import os
        e = os.remove(self.root_out + csvname)
        return e
