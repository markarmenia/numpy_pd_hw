import pandas as pd

# Mocked data since actual data was not provided
experiments_data = {
    'experiment_id': [1],
    'experiment_name': ['Experiment 1'],
    'property_name': ['channel'],
    'property_value': ['1']
}

plates_data = {
    'plate_id': [1, 2],
    'plate_name': ['Plate 1', 'Plate 2'],
    'experiment_id': [1, 1],
    'property_name': ['channel', 'channel'],
    'property_value': ['1', '1']
}

wells_data = {
    'well_id': [1, 2, 3, 4, 5, 6],
    'well_row': ['1', '1', '1', '1', '2', '2'],
    'well_column': ['1', '2', '3', '4', '1', '2'],
    'plate_id': [1, 1, 1, 1, 2, 2],
    'property_name': ['concentration', 'concentration', 'concentration', None, None, None],
    'property_value': ['1', '2', '3', None, None, None]
}

experiments = pd.DataFrame(experiments_data)
plates = pd.DataFrame(plates_data)
wells = pd.DataFrame(wells_data)

wells_pivot = wells.pivot_table(index=['well_id', 'well_row', 'well_column', 'plate_id'], 
                                columns='property_name', values='property_value', aggfunc='first').reset_index()

plates_pivot = plates.pivot_table(index=['plate_id'], 
                                  columns='property_name', values='property_value', aggfunc='first').reset_index()

experiments_pivot = experiments.pivot_table(index=['experiment_id'], 
                                            columns='property_name', values='property_value', aggfunc='first').reset_index()

result = pd.merge(wells_pivot, plates_pivot, on='plate_id', suffixes=('', '_plate'), how='left')

result = pd.merge(result, experiments_pivot, left_on='plate_id', right_on='experiment_id', suffixes=('', '_experiment'), how='left')

result['concentration_unit'] = result['concentration'].apply(lambda x: 'ul' if pd.notnull(x) else None)

if 'channel_plate' in result.columns and 'channel_experiment' in result.columns:
    result['channel'] = result['channel'].combine_first(result['channel_plate']).combine_first(result['channel_experiment'])
elif 'channel_experiment' in result.columns:
    result['channel'] = result['channel'].combine_first(result['channel_experiment'])
else:
    result['channel'] = result['channel'] 

columns_to_drop = ['plate_id', 'experiment_id', 'property_name']
existing_columns = [col for col in columns_to_drop if col in result.columns]
result = result.drop(columns=existing_columns)
result = result.drop([col for col in result.columns if '_plate' in col or '_experiment' in col], axis=1)

result.to_excel('result.xlsx', index=False)


