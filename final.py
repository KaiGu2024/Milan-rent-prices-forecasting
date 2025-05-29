import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV, Lasso
from scipy.stats import zscore

train = pd.read_csv("train.csv")
print(train.isnull().sum())
test = pd.read_csv("test.csv")

y = train['y']
print(f"Skewness: {skew(y):.4f}")
print(f"Kurtosis: {kurtosis(y):.4f}")
y_log = np.log1p(train['y']) 
print(f"Skewness: {skew(y_log):.4f}")
print(f"Kurtosis: {kurtosis(y_log):.4f}")

df = train.drop(columns=['y','w'])

## check missing values
print(train.isnull().sum())

## check availability high-frequency
df["availability"] = df["availability"].fillna("unknown").str.lower().str.strip()
df["is_available_now"] = df["availability"].apply(lambda x: 1 if x == "available" else 0)
df["available_date_raw"] = df["availability"].str.extract(r'available from ([\d]{1,2}/[\d]{1,2}/[\d]{4})')
df["available_date"] = pd.to_datetime(df["available_date_raw"], format="%d/%m/%Y", errors="coerce")

plt.figure(figsize=(10, 4))
sns.histplot(df["available_date"].dropna(), bins=50, kde=False)
plt.title("Distribution of Available From Dates")
plt.xlabel("Available Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## check desprition
description = df["description"].fillna("").astype(str).str.lower()
def tokenize(text):
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
token_set = set()
for desc in description:
    tokens = tokenize(desc)
    token_set.update(tokens)
token_list = sorted(token_set)
print(token_list)
print(f"共发现 {len(token_list)} 个唯一词项。")

## checj other_features
other_features = df["other_features"].fillna("").astype(str).str.lower()
def tokenize(text):
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
token_set = set()
for desc in other_features:
    tokens = tokenize(desc)
    token_set.update(tokens)
token_list = sorted(token_set)
print(token_list)
print(f"共发现 {len(token_list)} 个唯一词项。")

def preprocess(train, test):
    train['dataset'] = 'train'
    test['dataset'] = 'test'
    combined = pd.concat([train, test], axis=0)
    combined = pd.concat([train, test], keys=["train", "test"])

    # 统一处理字符串列，保证是 Series，填充、转小写、strip
    for col in ["contract_type", "availability", "conditions", "floor", "elevator", "energy_efficiency_class"]:
        combined[col] = combined[col].astype(str).fillna("unknown").str.lower().str.strip()

    # ========== 1. Contract Type ==========
    contract_dummies = pd.get_dummies(combined["contract_type"], prefix="contract", drop_first=True)
    combined = pd.concat([combined.drop(columns=["contract_type"]), contract_dummies], axis=1)

    # ========== 2. Availability ==========
    combined["is_available_now"] = combined["availability"].apply(lambda x: 1 if x == "available" else 0)
    combined["available_date_raw"] = combined["availability"].str.extract(r'available from ([\d]{1,2}/[\d]{1,2}/[\d]{4})')
    combined["available_date"] = pd.to_datetime(combined["available_date_raw"], format="%d/%m/%Y", errors="coerce")

    def bucket_availability(date):  # 画图后发现高频特征集中在2024年
        if pd.isnull(date): 
            return "unknown"
        elif date <= pd.Timestamp("2024-03-01"): 
            return "early"
        elif date <= pd.Timestamp("2024-06-01"): 
            return "spring_2024"
        elif date <= pd.Timestamp("2024-09-01"): 
            return "summer_2024"
        elif date <= pd.Timestamp("2024-12-31"): 
            return "fall_2024"
        else: 
            return "future"

    combined["availability_bucket"] = combined["available_date"].apply(bucket_availability)
    availability_dummies = pd.get_dummies(combined["availability_bucket"], prefix="avail", drop_first=True)
    combined = pd.concat([combined.drop(columns=["availability", "available_date_raw", "available_date", "availability_bucket"]), availability_dummies], axis=1)

    # ========== 3. Description: Extract Structured Info ==========
    kitchen_words = {'kitchen', 'kitchenette', 'diner', 'nook'}
    feature_words = {'habitable', 'open', 'semi', 'disabled'}
    stopwords = {'for', 'or', 'people', 'other', 'others', 'more', 'suitable', 'room', 'rooms'}

    # 所有可能的特征词汇表（要 dummy 的）
    all_features = sorted(kitchen_words.union(feature_words))
    parsed_data = []

    for desc in combined['description'].astype(str).str.lower():
        tokens = re.sub(r"[^\w\s]", " ", desc).split()

        total_room = 0
        bedroom = 0
        bathroom = 0
        dummies = {f'is_{w}': 0 for w in all_features}

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 首词 + 括号，作为总房间数
            if i == 0 and token.isdigit() and '(' in desc:
                total_room = int(token)
                i += 1
                continue

            # 数字 + 类型
            if token.isdigit() and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token in {'bedroom', 'bedrooms'}:
                    bedroom += int(token)
                    i += 2
                    continue
                elif next_token in {'bathroom', 'bathrooms'}:
                    bathroom += int(token)
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            # 单词特征（忽略 stopwords）
            elif token not in stopwords:
                if token in all_features:
                    dummies[f'is_{token}'] = 1
            i += 1

        # fallback：若 total_room 仍为 0，使用 bedroom + bathroom 合计
        if total_room == 0:
            total_room = bedroom + bathroom

        parsed_data.append({
            'total_room': total_room,
            'bedroom': bedroom,
            'bathroom': bathroom,
            **dummies
        })

    result_df = pd.DataFrame(parsed_data, index=combined.index)
    combined = pd.concat([combined.drop(columns = ['description']), result_df], axis=1)
    print("Columns after description parse:", combined.columns)

    # other_features
    # 处理 other_features 列
    other_feats = combined['other_features'].fillna('').str.lower().str.replace(r'[^\w\s]', ' ', regex=True).str.split()

    structure_features = {
        'attic', 'balcony', 'balconies', 'cellar', 'closet', 'court', 'fireplace', 'frames',
    '    garden', 'pool', 'tavern', 'tennis', 'terrace', 'window'
    }
    material_features = {'metal', 'wood', 'glass', 'pvc', 'pvcdouble', 'pvcexposure'}
    security_features = {'alarm', 'concierge', 'entryphone', 'reception', 'reception1', 'security', 'gate', 'video', 'door'}
    media_features = {'fiber', 'optic', 'tv', 'satellite', 'dish'}
    orientation_features = {'east', 'west', 'south', 'north', 'exposure', 'internal', 'external'}
    furniture_features = {'furnished', 'partially', 'only', 'single', 'double', 'triple', 'full', 'half'}
    bath_features = {'hydromassage'}
    system_features = {'centralized', 'electric', 'system'}
    stopwords = {'in', 'and', 'day', 'with'}

    # 全部合法特征
    all_features = (
        structure_features | material_features | security_features | media_features |
        orientation_features | furniture_features | bath_features | system_features
    )

    # 初始化空 DataFrame 用于哑变量
    for feat in sorted(all_features):
        combined[f'feat_{feat}'] = 0

    # 遍历行
    for idx, tokens in other_feats.items():
        for token in tokens:
            if token in all_features:
                combined.at[idx, f'feat_{token}'] = 1

    # 删除原始列
    combined.drop(columns=['other_features'], inplace=True)

    # ========== 6. Floor, Elevator, Condition ==========
    condition_dummies = pd.get_dummies(combined["conditions"], prefix="cond", drop_first=True)
    combined = pd.concat([combined.drop(columns=["conditions"]), condition_dummies], axis=1)

    combined["floor"] = combined["floor"].replace({
        "ground floor": "ground", 
        "semi-basement": "basement", 
        "mezzanine": "mezzanine"
    }).fillna("unknown")
    floor_dummies = pd.get_dummies(combined["floor"], prefix="floor", drop_first=True)
    combined = pd.concat([combined.drop(columns=["floor"]), floor_dummies], axis=1)

    elevator_dummies = pd.get_dummies(combined["elevator"], prefix="elevator", drop_first=True)
    combined = pd.concat([combined.drop(columns=["elevator"]), elevator_dummies], axis=1)

    # ========== 7. Energy Efficiency ==========
    combined["energy_efficiency_class"] = combined["energy_efficiency_class"].replace({",": np.nan, "nan": np.nan}).fillna("unknown")
    eff_dummies = pd.get_dummies(combined["energy_efficiency_class"], prefix="eff", drop_first=True)
    combined = pd.concat([combined.drop(columns=["energy_efficiency_class"]), eff_dummies], axis=1)

    ## condominium fees
    combined['condominium_fees'] = combined.groupby('zone')['condominium_fees'].transform(
        lambda x: x.fillna(x.median())
    )
    combined['condominium_fees'] = combined['condominium_fees'].fillna(combined['condominium_fees'].median())

    ## zone  length = 132
    '''
    min_samples = int(0.01 * len(combined))  

    zone_counts = combined['zone'].value_counts()
    combined['zone_group'] = combined['zone'].where(
        combined['zone'].isin(zone_counts[zone_counts >= min_samples].index), 
    '    other'
    )

    n_categories = combined['zone_group'].nunique()
    print(f" {n_categories}") # 40
   '''
    zone_dummies = pd.get_dummies(
        combined['zone'],
        prefix='zone',
        drop_first=True,  
    )

    combined = pd.concat([combined, zone_dummies], axis=1)
    combined = combined.drop(['zone'], axis=1)

        # ========== 9. Feature Engineering ==========
    combined["log_sqm"] = np.log1p(combined["square_meters"])
    combined["sqm_sq"] = combined["square_meters"] ** 2
    combined["sqrt_sqm"] = np.sqrt(combined["square_meters"])
    combined["log_condo"] = np.log1p(combined["condominium_fees"])
    combined["log_rooms"] = np.log1p(combined["total_room"])
    combined["sqm_x_rooms"] = combined["square_meters"] * combined["total_room"]
    combined["sqm_x_fees"] = combined["square_meters"] * combined["condominium_fees"]
    combined["room_density"] = combined["bedroom"] / (combined["total_room"] + 1e-3)

    combined["rooms_x_fees"] = combined["total_room"] * combined["condominium_fees"]
    combined["sqm_x_bedrooms"] = combined["square_meters"] * combined["bedroom"]
    
    '''
    # ========== 10. Outlier and Leverage Filtering ==========
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = combined[numeric_cols].apply(zscore)
    combined = combined[(np.abs(z_scores) < 5).all(axis=1)]  # remove outliers

    # ========== 11. Scale/transform ==========
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [col for col in numeric_cols if combined[col].nunique() <= 2]
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]

    for col in scale_cols:
        combined[col] = np.log1p(combined[col].clip(lower=0))
    '''

    # ========== 12. Final Split ==========
    df = combined[combined['dataset'] == 'train'].copy()
    test_processed = combined[combined['dataset'] == 'test'].copy()
    combined.drop(columns=['dataset'], inplace=True) 

    return df, test_processed

df, test = preprocess(df,test)
print(df.columns.tolist())
df = df.drop(columns=['dataset'])
test = test.drop(columns=['dataset'])

# baseline: lasso
X = df
lasso_cv = LassoCV(cv=5, random_state=42).fit(X, y_log)
best_alpha = lasso_cv.alpha_
print("Best alpha from LassoCV:", best_alpha)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
y_true = np.expm1(y_log)

for train_idx, val_idx in cv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_val)
    y_pred_real = np.expm1(y_pred_log)
    y_val_real = np.expm1(y_val_log)

    mae = mean_absolute_error(y_val_real, y_pred_real)
    mae_scores.append(mae)

print(f"Mean MAE (original scale): {np.mean(mae_scores):.4f}")
print(f"Std of MAE: {np.std(mae_scores):.4f}")

# gradien boosting
X = df
y_true = np.expm1(y_log)  
# print(X)
xgb_params = {
    'n_estimators': [1000, 2000, 3000],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.001, 0.01, 0.05],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_cv = GridSearchCV(
    estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid=xgb_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

xgb_cv.fit(X, y_log)
print("Best XGBoost params:", xgb_cv.best_params_)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

for train_idx, val_idx in cv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    model = XGBRegressor(
        **xgb_cv.best_params_,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train_log)
    
    y_pred_log = model.predict(X_val)
    y_pred_real = np.expm1(y_pred_log)
    y_val_real = np.expm1(y_val_log)
    
    mae = mean_absolute_error(y_val_real, y_pred_real)
    mae_scores.append(mae)

print("\nXGBoost Performance:")
print(f"Mean MAE (original scale): {np.mean(mae_scores):.4f}")
print(f"Std of MAE: {np.std(mae_scores):.4f}")

# stack
## lightBGM
lgb_params = {
    'n_estimators': [300, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.005, 0.01, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgb_cv = GridSearchCV(
    estimator=lgb.LGBMRegressor(random_state=42),
    param_grid=lgb_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
lgb_cv.fit(X, y_log)
print("Best LightGBM params:", lgb_cv.best_params_)
## catboost
cat_params = {
    'iterations': [300, 500, 1000],
    'depth': [3, 5, 7],
    'learning_rate': [0.005, 0.01, 0.05]
}

cat_cv = GridSearchCV(
    estimator=CatBoostRegressor(random_seed=42, verbose=0),
    param_grid=cat_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
cat_cv.fit(X, y_log)
print("Best CatBoost params:", cat_cv.best_params_)

# 4. 用最佳参数初始化模型
xgb_best = XGBRegressor(random_state=42, objective='reg:squarederror', **xgb_cv.best_params_)
lgb_best = lgb.LGBMRegressor(random_state=42, **lgb_cv.best_params_)
cat_best = CatBoostRegressor(random_seed=42, verbose=0, **cat_cv.best_params_)

# 5. Stacking模型（最终学习器用岭回归）
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_best), ('lgb', lgb_best), ('cat', cat_best)],
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

# 6. KFold CV计算MAE（真实尺度）
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

for train_idx, val_idx in cv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

    stacking_model.fit(X_train, y_train_log)
    y_pred_log = stacking_model.predict(X_val)

    y_pred_real = np.expm1(y_pred_log)
    y_val_real = np.expm1(y_val_log)

    mae = mean_absolute_error(y_val_real, y_pred_real)
    mae_scores.append(mae)

print(f"Stacking Model Mean MAE (original scale): {np.mean(mae_scores):.4f}")
print(f"Stacking Model Std MAE: {np.std(mae_scores):.4f}")


# generate prediction
'''
train_features = df.drop(columns=['w', 'y', 'log_y']).columns  
df_var = selector.fit_transform(df[train_features])  
cols_var = train_features[selector.get_support()]   
test_var = selector.transform(test[train_features])  
test_var = pd.DataFrame(test_var, columns=cols_var, index=test.index)
test_uncorr = test_var.drop(columns=[col for col in dropped_cols if col in test_var.columns])
test_scaled = scaler.transform(test_uncorr)
test_pca = pca_final.transform(test_scaled)
test_pca = pd.DataFrame(test_pca, columns=[f'PC{i+1}' for i in range(n_components)], index=test.index)
'''

test_pred_log = stacking_model.predict(test)
test_pred = np.expm1(test_pred_log)

with open("output.txt", "w") as f:
    for value in test_pred:
        f.write(f"{value:.6f}\n")  


## PCA picture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设combined是包含所有数值特征的DataFrame，已经做了适当的数值编码和缺失值处理

# 选取数值型特征（如果包含分类需先编码）
numeric_features = X.select_dtypes(include=[np.number]).fillna(0)

# 标准化（PCA通常建议先标准化）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_features)
print(numeric_features)
# 运行PCA，n_components设置为全部
pca = PCA(n_components=X_scaled.shape[1])
pca.fit(X_scaled)

# 计算累计解释方差比例
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var_exp)+1), cum_var_exp, marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# 结果图
stacking_model.fit(X, y_log)
y_pred_log = stacking_model.predict(X)
y_pred_real = np.expm1(y_pred_log)
y_real = np.expm1(y_log)

# 1. 预测值 vs 真实值散点图
plt.figure(figsize=(6,6))
plt.scatter(y_real, y_pred_real, alpha=0.4)
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
plt.xlabel("True Rent Price")
plt.ylabel("Predicted Rent Price")
plt.title("Predicted vs True Rent Price (Stacking Model)")
plt.tight_layout()
plt.show()

# 2. 残差分布图
residuals = y_real - y_pred_real
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=50, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (True - Predicted)")
plt.title("Residuals Distribution (Stacking Model)")
plt.tight_layout()
plt.show()

# 3. XGBoost特征重要性条形图
# 取stacking模型中的xgb基学习器
xgb_model = stacking_model.named_estimators_['xgb']

# 注意：xgb_model 是 XGBRegressor，使用get_booster获取底层Booster
importance = xgb_model.get_booster().get_score(importance_type='gain')
importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(8,6))
sns.barplot(x=list(importance.values())[:20], y=list(importance.keys())[:20])
plt.xlabel("Feature Importance (Gain)")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importances in XGBoost")
plt.tight_layout()
plt.show()