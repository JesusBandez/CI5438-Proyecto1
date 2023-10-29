import pandas as pd
import re
class DataCleaner():
    def __init__(self, file_path) -> None:
        self._df = pd.read_csv(file_path)

    def clean_dataset(self):
        self._filter_columns()
        self._treat_missing_data()
        self._format_data()
        self._normalize_data()
        self._convert_categoric_to_dummies()
        print(self._df)

        return self._df
    
    def _filter_columns(self):
        """Quita del dataframe todas las columnas excepto las especificadas
         en la variable columns_to_keep"""
        self._columns_to_keep = [
            "Make", "Year",
            "Kilometer", "Fuel Type",
            "Transmission", "Owner", 
            "Engine", 
            "Seating Capacity", "Fuel Tank Capacity"
        ]
        # TODO:
        # No creo que numero de duenio valga la pena incluirlo,
        # despues de todo lo que importa es cuanto timepo ha pasado
        # desde que se fabrico el vehiculo, no la cantidad de gente que lo
        # ha comprado

        # Incluimos las carateresticas del motor para variar un poco
        self._df = self._df[self._columns_to_keep]

    def _treat_missing_data(self):
        # TODO:
        # Este enfoque elimina las filas con datos faltantes.
        # El dataset tiene 2056 filas al inicio. Eliminar los faltantes
        # deja un total de 1937 filas.
        # Los datos eliminados corresponden a un 5.69% del dataset.
        self._df = self._df.dropna()

    def _format_data(self):
        """Aplica transformaciones sobre las columnas para conseguir 
            los valores numericos"""
        
        # Obtener el numero de cc del motor
        def format_engine(data):
            res = re.search("[0-9]+",data)
            return res.group(0) if res else None
        self._df["Engine"] = self._df["Engine"].map(format_engine)
        
    def _normalize_data(self):
        "Normaliza las columnas que sean normalizables"
        colums_to_normalize = [
            "Year", "Kilometer", "Engine", "Seating Capacity", "Fuel Tank Capacity"
        ]

        for col in colums_to_normalize:
            self._df[col] = pd.to_numeric(self._df[col])
            min_in_col = self._df[col].min()
            max_in_col = self._df[col].max()
            normalized_serie = self._df[col].\
                map(lambda x: (x-min_in_col)/(max_in_col-min_in_col))
            self._df[col] = normalized_serie
        # TODO: Uno de los kilometros es muy alto.
        # TODO: Utilizar graficos de frecuencia para detectar datos que valga la pena eliminar
        

    def _convert_categoric_to_dummies(self):
        "Divide las variables categoricas en variables binarias dummies"
        categoric_columns = [
            "Make", "Fuel Type", "Transmission", "Owner"
        ]
        for col in categoric_columns:
            for category in self._df[col].unique():
                self._df[f"{col}-{category}"] = 0
                self._df.loc[self._df[col] == category, f"{col}-{category}"] = 1
            self._df = self._df.drop(col,axis=1)
            

if __name__ == "__main__":
    cleaner = DataCleaner("CarDekho.csv")
    cleaner.clean_dataset().to_csv("CleanedCarDekho.csv", index=False)