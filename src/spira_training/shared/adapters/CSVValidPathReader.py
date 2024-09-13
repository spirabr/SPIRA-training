import pandas as pd
from typing import List

from src.spira_training.shared.core.models.valid_path import ValidPath
from src.spira_training.shared.ports.valid_path_reader import ValidPathReader


class CSVValidPathReader(ValidPathReader):
    def read_valid_paths(self, csv_path: ValidPath) -> List[ValidPath]:
        df = pd.read_csv(str(csv_path), header=None, sep=",")
        self.validate_dataframe(df)

        paths = df[0].tolist()
        valid_paths = ValidPath.from_list(paths)
        return valid_paths

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty or df[0].isnull().any() or (df[0].str.strip() == '').any():
            raise ValueError("A primeira coluna do CSV contém valores nulos ou vazios.")
