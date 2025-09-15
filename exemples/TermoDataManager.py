import polars as pl
from src.DataManager import DataManager

class TermoDataManager(DataManager):
    def __init__(self, path_to_data:str=None, data_files_name:list[str]=None, verbose:bool=False) -> None:
        super().__init__(path_to_data, data_files_name, load_data=False, verbose=verbose)
        # specific load_data that uses tab as separator
        self.data = self.load_data(separator='\t')

    def _power_preprocess(self, power_df):
        # removing lower power 
        power_threshold_dict = {'Power6':0.01, 'Power7': 0.04,'Power8': 0.01,'Power9': 0.01,'Power10': 0.04}
        # if the column value is ge than the threshold then return the same value
        # otherwise 0, 
        power_df = power_df.with_columns([
            pl.when(pl.col(col) >= thr).then(pl.col(col)).otherwise(0).alias(col)
            for col, thr in power_threshold_dict.items()
        ])

        power_df = power_df.with_columns([
            pl.sum_horizontal(pl.all().alias("power_sum"))
        ])

        return power_df
    
    def get_data_in_out(self, verbose:bool=False) -> tuple[pl.DataFrame, pl.DataFrame]:
        '''
        Returns input and output data for the thermal system example.
        '''
        data1 = self.data[self.data_files_name[0]]
        data2 = self.data[self.data_files_name[1]]
        data3 = self.data[self.data_files_name[2]]
        data4 = self.data[self.data_files_name[3]]
        
        # Obtain the power columns
        power_columns = [col for col in data4.columns if col.startswith("Power")]
        power_df = data4[power_columns]
        power_df = self._power_preprocess(power_df)

        data_in = pl.DataFrame([data1["TentHT"], data3["Tamb"], data2["NumVentOn"]])
        data_in = pl.concat([data_in, power_df], how="horizontal")
        data_out = pl.DataFrame(data1["TsaidaHT"])
        
        if self.verbose or verbose:
            print("\n========== before min_max normalization ==========")
            self.print_input_output_range(data_in, data_out)
        
        # we'll apply normalization to all but NumVentOn data
        num_vent_on = data_in[["NumVentOn"]]
        data_in_norm = self.min_max_normalization(data_in.drop("NumVentOn"))
        data_in = pl.concat([data_in_norm, num_vent_on], how="horizontal")
        data_out = self.min_max_normalization(data_out)

        if self.verbose or verbose:
            print("\n========== after min_max normalization ==========")
            self.print_input_output_range(data_in, data_out)        

        return data_in, data_out   