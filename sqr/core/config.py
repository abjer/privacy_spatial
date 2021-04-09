dummy_compute = False
dummy_mun_codes = [147,151,153,165,400]
dummy_mun_codes_4char = [f'0{m}' for m in dummy_mun_codes]

cell_label = 'DDKNm100'

years_num = list(range(2010,2020))
years = [str(y) for y in years_num]
years_hh = [f'{y}_hh' for y in years]
years_pers = [f'{y}_pers' for y in years]

mean_cols = ['mean_pers', 'mean_hh']
minimum_cols= ['minimum_pers', 'minimum_hh']