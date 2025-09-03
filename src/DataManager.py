import os
import polars as pl


class DataManager:
    def __init__(self, path_to_data:str=None, data_files_name:list[str]=None, load_data:bool=True, verbose:bool=False) -> None:
        self.path_to_data = path_to_data
        self.data_files_name = data_files_name
        self.data = self.load_data() if load_data else None
        self.verbose = verbose

    def load_data(self, separator:str=',') -> pl.DataFrame:
        data = {}
        if self.verbose:
            print("\t\t\t =loaded data=:")
        for file_name in self.data_files_name:
            file_path = os.path.join(self.path_to_data, file_name)
            try:
                data[file_name] = pl.read_csv(file_path, separator=separator)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                raise e
            if self.verbose:
                print(f"=========== {file_name} ===========:\n", data[file_name].head())
                print("has Null:", data[file_name])
        return data

    def get_data_in_out(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        ''' 
        This method is intended to be overridden in subclasses.
        Here you'll create how to extract input and output data from the loaded datasets.
        It should return two DataFrames: one for input data and another for output data.
        '''
        raise NotImplementedError("This method should be implemented in a subclass. See TermoDataManager for an example.")
    
    def split_data(self, data_in:pl.DataFrame, data_out:pl.DataFrame, train_ratio:float=0.8) -> tuple:
        total_samples = data_in.height
        train_size = int(total_samples * train_ratio)

        X_train = data_in[:train_size]
        y_train = data_out[:train_size]
        X_val = data_in[train_size:]
        y_val = data_out[train_size:]

        return X_train, y_train, X_val, y_val
                
    def save_output(self, output:pl.DataFrame, file_name:str='output.csv', verbose=True) -> None:
        output_path = os.path.join(self.path_to_data, file_name)
        output.write_csv(output_path)
        if self.verbose or verbose:
            print(f"Output saved to {output_path}")

    def min_max_normalization(self, df:pl.DataFrame):
        return df.select((pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min())) 
    
    def dataframe_range(self, df:pl.DataFrame):
        df_range_list = []
        for col in df.columns:
            df_range_list.append((df[col].min(), df[col].max()))
        return df_range_list
    
    def print_input_output_range(self, data_in:pl.DataFrame, data_out:pl.DataFrame) -> None:
        input_range = self.dataframe_range(data_in)
        output_range = self.dataframe_range(data_out)

        print(f"=========== data_in ===========:\n", data_in.head())
        print("input range:")
        for i, col in enumerate(data_in.columns):
            col = col.strip()
            print(col, end=': ')
            print(f"Min: {input_range[i][0]:.2f} - Max:{input_range[i][1]:.2f}")    
        
        print(f"\n=========== data_out ===========:\n", data_out.head())
        print("output range:")
        for i, col in enumerate(data_out.columns):
            col = col.strip()
            print(col, end=': ')
            print(f"Min: {output_range[i][0]:.2f} - Max:{output_range[i][1]:.2f}") 
