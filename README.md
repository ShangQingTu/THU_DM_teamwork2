# THU_DM_teamwork2
Code for the Hospital Readmission Clustering Analysis task.

## Data Preprocessing
### Id to description mapping
IDs_mapping.csv文件里的数据去除含NaN的行之后，把 `admission_type_id`， `discharge_disposition_id`，  `admission_source_id` 三种id到description的映射保存在一个字典里<br><br>用法：
```python
import json

with open('data/id2desc.json', 'r') as f:
    id2desc = json.load(f)

id2desc['admission_type_id'].get('0', None) #return None
id2desc['admission_type_id'].get('1', None) #return 'Emergency'
id2desc['admission_source_id'].get('1', None) #return ' Physician Referral'
id2desc['discharge_disposition_id'].get('1', None) #return 'Discharged to home'
```

### Data Cleaning
1. 所有属性都是Categorical型属性，不存在outlier的问题
2. 删除缺失过半的`weight`，`payer_code`，`medical_specialty`属性
3. 删除每个病人独有的id属性`encounter_id`，`patient_nbr`
4. 按题目要求删除`readmitted`属性
5. 对于所有含有 ? 的行进行了删除

### Feature Selection
删除所有entropy小于0.1(所有样本在该属性上几乎或者完全相同)的属性，包括：
 `repaglinide`，
 `nateglinide`，
 `chlorpropamide`，
 `acetohexamide`，
 `tolbutamide`，
 `acarbose`，
 `miglitol`，
 `troglitazone`，
 `tolazamide`，
 `examide`，
 `citoglipton`，
 `glyburide-metformin`，
 `glipizide-metformin`，
 `glimepiride-pioglitazone`，
 `metformin-rosiglitazone`，
 `metformin-pioglitazone`
 
 ### Feature Transform
 将剩下的所有特征分为三类：nominal、ordinal、numerical，其中
 - nominal: `race`, `gender`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`,`change`, `diabetesMed`,`metformin`, `glimepiride`, `glipizide`, `glyburide`, `pioglitazone`,
       `rosiglitazone`, `insulin`
 - ordinal: `age`, `diag_1`,
       `diag_2`, `diag_3`, `max_glu_serum`, `A1Cresult`
 - numerical: `time_in_hospital`,
       `num_lab_procedures`, `num_procedures`, `num_medications`,
       `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`

对于nominal属性，在使用时可以通过OneHot Encoding将其变为numerical属性，或者使用能够计算含有nominal属性数据的distance metric<br><br>
对于ordinal属性，`diag_123`使用的是ICD9的疾病分类编码，三次诊断涵盖的种类不完全重合，因此将三次诊断的所有数据合并使用LabelEncoder进行编码，`max_glu_serum`, `A1Cresult`为了反映正确的序关系，手动对其进行编码，后续可以直接作为numerical属性使用<br><br>
对于numerical属性不做处理