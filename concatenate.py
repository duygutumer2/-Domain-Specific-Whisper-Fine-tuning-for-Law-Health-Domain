from datasets import load_from_disk

data1 = load_from_disk('ar-dataset-med')
data2 = load_from_disk('cs_dataset_med')
data3 = load_from_disk('de_dataset_med')
data4 = load_from_disk('el_dataset_med')
data5 = load_from_disk('en_dataset_med')
data6 = load_from_disk('es_dataset_med')
data7 = load_from_disk('fa_dataset_med')
data8 = load_from_disk('fr_dataset_med')
data9 = load_from_disk('he_dataset_med')
data10 = load_from_disk('hi_dataset_med')
data11 = load_from_disk('id_dataset_med')
data12 = load_from_disk('it_dataset_med')
data13 = load_from_disk('ja_dataset_med')
data14 = load_from_disk('ko_dataset_med')
data15 = load_from_disk('nl_health_dataset')
data16 = load_from_disk('pl_dataset_med')
data17 = load_from_disk('pt_dataset_med')
data18 = load_from_disk('ro_dataset_med')
data19 = load_from_disk('ru_dataset_med')
data20 = load_from_disk('tr_health_dataset')
data21 = load_from_disk('uk_health_dataset')
data22 = load_from_disk('vi_health_dataset')
data23 = load_from_disk('zh_health_dataset')

med = load_from_disk("med-dataset")
print(med)
from datasets import concatenate_datasets
"""
# Concatenate the two Datasets
combined_dataset = concatenate_datasets([data1, data2,data3, data4, data5, data6, data7, data8, data9, data10, data11,
                                         data12, data13, data14, data15, data16,
                                        data17, data18, data19, data20, data21, data22, data23])

print(combined_dataset)

combined_dataset.save_to_disk('med-dataset')"""