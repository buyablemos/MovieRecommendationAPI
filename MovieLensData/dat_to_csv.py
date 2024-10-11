from io import StringIO

import pandas as pd
import chardet


movies_path = 'movies.dat'
ratings_path = 'ratings.dat'
users_path = 'users.dat'

for file_path in [movies_path, ratings_path, users_path]:

    with open(file_path, 'rb') as file:
        raw_data = file.read()

    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

    # Dekodowanie i przetwarzanie

    decoded_contents = raw_data.decode(encoding)
    decoded_contents = decoded_contents.replace('::', ';')

    data = pd.read_csv(StringIO(decoded_contents), sep=';', engine='python')

    print(data.head())

    output_csv_path = file_path.replace('.dat','.csv')
    data.to_csv(output_csv_path, index=False,sep=';')

    print(f'Plik zapisany jako {output_csv_path}')

file_path=users_path.replace('.dat','.csv')
with open(file_path, 'rb') as file:
    data = pd.read_csv(file, sep=';')
    data['zip-code'] = data['zip-code'].astype(str).str.replace('-', '', regex=False)
    output_csv_path = file_path.replace('.dat','.csv')
    data.to_csv(output_csv_path, index=False,sep=';')

    print(f'Plik zapisany jako {output_csv_path}')

