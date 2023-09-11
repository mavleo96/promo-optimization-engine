# !pip install azure-storage-blob
# !pip install python-dotenv

import os
from dotenv import load_dotenv
from setup_utils import fetch_data, load_data, create_time_index
from datetime import datetime
import pandas as pd
import numpy as np

load_dotenv()
fetch_data(CONNECTION_STRING)

(
    brand_mapping_backup,
    macro_data_backup,
    brand_constraint_backup,
    pack_constraint_backup,
    segment_constraint_backup,
    sales_data_backup,
    volume_variation_constraint_backup,
) = load_data()

(
    macro_data_backup,
    sales_data_backup,
) = create_time_index([macro_data_backup, sales_data_backup])

from sklearn.metrics import make_scorer, r2_score

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
import tensorflow as tf

brand_mapping = brand_mapping_backup.copy(deep=True)
macro_data = macro_data_backup.copy(deep=True)
brand_constraint = brand_constraint_backup.copy(deep=True)
pack_constraint = pack_constraint_backup.copy(deep=True)
segment_constraint = segment_constraint_backup.copy(deep=True)
sales_data = sales_data_backup.copy(deep=True)
volume_variation_constraint = volume_variation_constraint_backup.copy(deep=True)

sales_index = sales_data.index.unique()
macro_data = macro_data.loc[sales_index].sort_index()
covid = pd.Series([1 if (i<=datetime(2020,5,1) and i>=datetime(2020,3,1)) else 0 for i in macro_data.index], index=sales_index, name="covid")
macro_data = macro_data.join(covid)
constraints_dict = {
    "brand" : brand_constraint,
    "pack" : pack_constraint,
    "segment" : segment_constraint,
    "volume_variation" : volume_variation_constraint
}
temp_data = sales_data[sales_data.gto.isna()].reset_index()
temp_data["month"] = temp_data.date.dt.month
temp_data["year"] = temp_data.date.dt.year
temp_data = temp_data.fillna(10000)
temp_data = temp_data.merge(brand_mapping)

master_mapping = temp_data[["sku", "pack", "brand", "segment"]].drop_duplicates().reset_index(drop=True)
def _create_encodings(master_map):

    def label_encoder(series):
        unique_values = series.sort_values().unique()
        unique_count =  series.nunique()

        return dict(zip(unique_values, range(len(unique_values))))

    def mapper(col_val, col_key="sku"):

        df = master_map[[col_key, col_val]].drop_duplicates()
        df.loc[:,col_val] = df[col_val].map(label_dict[col_val])
        df.loc[:,col_key] = df[col_key].map(label_dict[col_key])

        return df.set_index(col_key).to_dict()[col_val]

    label_dict = {col:label_encoder(master_map[col]) for col in master_map.columns}
    mapper_dict = {col:mapper(col) for col in master_map.columns if col!="sku"}

    return {"label_dict" : label_dict, "mapper_dict" : mapper_dict}

final_encodings = _create_encodings(master_mapping)
total_sku_list = np.sort(sales_data.sku.unique()).tolist()
target_sku_list = list(final_encodings["label_dict"]["sku"].keys())
non_target_sku_list = [i for i  in total_sku_list if i not in final_encodings["label_dict"]["sku"]]
sku_index_order = [*target_sku_list, *non_target_sku_list]
def _constraint_tensor_generate(constraint, encoding, key):

    encoding_length = max(encoding["label_dict"][key].values())+1
    constraint = constraint.copy(deep=True)
    constraint = constraint.replace(encoding["label_dict"][key]).sort_values(["month", key])
    constraint = constraint.groupby(["month", key]).max_discount.sum().sort_index().unstack(1)

    constraint = pd.DataFrame(columns=pd.Index(range(0,encoding_length), dtype='int64', name="brand"), index=pd.Index(range(6,8), dtype='int64')).fillna(constraint).fillna(0.0).to_numpy()

    return constraint

brand_constraint_tensor = _constraint_tensor_generate(constraints_dict["brand"], final_encodings, "brand")
pack_constraint_tensor = _constraint_tensor_generate(constraints_dict["pack"], final_encodings, "pack")
segment_constraint_tensor = _constraint_tensor_generate(constraints_dict["segment"], final_encodings, "segment")
macro_data = macro_data.loc[sales_index].sort_index()
macro_data = (macro_data/macro_data.mean()-1).copy(deep=True)
macro_data = macro_data.astype(np.float64).values
macro_data = np.expand_dims(macro_data, 1)
nr_data = (
    sales_data
    .reset_index()
    .groupby(["date", "sku"])
    .net_revenue.sum()
    .sort_index()
    .unstack(1)
    [sku_index_order]
    .clip(0.0, None)
    .fillna(0.0)
    .astype(np.float64)
    .values
)
nr_data_mask = (
    sales_data
    .reset_index()
    .groupby(["date", "sku"])
    .net_revenue.sum()
    .sort_index()
    .unstack(1)
    [sku_index_order]
    .applymap(lambda x: x if x>=0 else np.nan)
    .notna()
    .astype(np.float64)
    .values
)

nr_shifted = (
    sales_data
    .reset_index()
    .groupby(["date", "sku"])
    .net_revenue.sum()
    .sort_index()
    .unstack(1)
    [sku_index_order]
    .applymap(lambda x: x if x>=0 else np.nan)
    .clip(0.0, None)
    .shift(1)
    .fillna(method="bfill")
    .fillna(0.0)
    .astype(np.float64)
    .values
)

volume_data = (
    sales_data
    .reset_index()
    .groupby(["date", "sku"])
    .volume.sum()
    .sort_index()
    .unstack(1)
    [sku_index_order]
    .clip(0.0, None)
    .fillna(0.0)
    .astype(np.float64)
    .values
)


discount_data = (
    sales_data
    .reset_index()
    .groupby(["date", "sku"])[["promotional_discount", "other_discounts"]].sum()
    .sort_index()
    .stack()
    .unstack(1)
    [sku_index_order]
    .fillna(0.0)
    .clip(None, 0)
)
discount_data = np.swapaxes(discount_data.astype(np.float64).values.reshape(55,2,discount_data.shape[1]), 1, 2)
/tmp/ipykernel_6215/3050755633.py:22: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
  .applymap(lambda x: x if x>=0 else np.nan)
/tmp/ipykernel_6215/3050755633.py:36: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
  .applymap(lambda x: x if x>=0 else np.nan)
/tmp/ipykernel_6215/3050755633.py:29: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  sales_data
scaler = nr_data.mean()
vol_scaler = volume_data.mean()

nr_data = nr_data/scaler
discount_data = discount_data/scaler
nr_shifted = nr_shifted/scaler
brand_constraint_tensor = brand_constraint_tensor/scaler
pack_constraint_tensor = pack_constraint_tensor/scaler
segment_constraint_tensor = segment_constraint_tensor/scaler


volume_data = volume_data/vol_scaler
time_index_array = np.expand_dims(np.arange(1, macro_data.shape[0]+1), 1)/100
tf.compat.v1.reset_default_graph()
tf.compat.v1.enable_eager_execution()

y = tf.constant(nr_data, dtype=tf.float64)
y_mask = tf.constant(nr_data_mask, dtype=tf.float64)

discounts = tf.constant(discount_data, dtype=tf.float64)
mixed_effect = tf.constant(macro_data, dtype=tf.float64)
time_index = tf.constant(np.expand_dims(np.arange(1, macro_data.shape[0]+1), 1), dtype=tf.float64)
shifted_nr = tf.constant(nr_shifted, dtype=tf.float64)
y_vol = tf.constant(volume_data, dtype=tf.float64)

val_splitter_ = tf.constant(5, dtype=tf.int32)
val_splitter = 5 #if val_splitter_ == 5 else 2

initial_discount_var = tf.slice(discounts, begin=[0,0,0], size=[2,-1,2])

mixed_effect_var = tf.slice(mixed_effect, begin=[0,0,0], size=[2,-1,-1])
time_index_var = tf.slice(time_index, begin=[0,0], size=[2,-1])
y_mask_var = tf.slice(y_mask, begin=[0,0], size=[2,-1])
2023-09-10 19:51:41.327571: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:268] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

#Y
y = tf.compat.v1.placeholder(dtype=tf.float64, name="nr_actual")
y_mask = tf.compat.v1.placeholder(dtype=tf.float64, name="nr_mask")

# X
discounts = tf.compat.v1.placeholder(dtype=tf.float64, name="discounts")
mixed_effect = tf.compat.v1.placeholder(dtype=tf.float64, name="mixed_effects")
time_index = tf.compat.v1.placeholder(dtype=tf.float64, name="time_index")
# shifted_nr = tf.compat.v1.placeholder(dtype=tf.float64, name="shifted_nr")
y_vol = tf.compat.v1.placeholder(dtype=tf.float64, name="volume_actual")

val_splitter_ = tf.compat.v1.placeholder(dtype=tf.int32)
val_splitter = 5 #if val_splitter_ == 5 else 2

initial_discount_var = tf.compat.v1.placeholder(dtype=tf.float64, name="initial_discount_submit")

mixed_effect_var = tf.compat.v1.placeholder(dtype=tf.float64, name="mixed_effect_submit")
time_index_var = tf.compat.v1.placeholder(dtype=tf.float64, name="time_index_submit")
y_mask_var = tf.compat.v1.placeholder(dtype=tf.float64, name="y_mask_submit")
dim_size = (1,nr_data.shape[1])
me_size = macro_data.shape[-1]

baseline_intercept = tf.Variable(np.expand_dims((nr_data.mean(0)*0.3), 0), dtype=tf.float64)

baseline_slope1_global = tf.Variable(np.full((1,1), 0.1), dtype=tf.float64)
baseline_slope1_hier = tf.Variable(np.full(dim_size, 0.1), dtype=tf.float64)

mixed_effect_mult_global = tf.Variable(np.random.normal(loc=0, size=(1, 1, me_size)), dtype=tf.float64)
mixed_effect_mult_hier = tf.Variable(np.random.normal(loc=0, size=(*dim_size, me_size)), dtype=tf.float64)

discount_slope_global = tf.math.sigmoid(tf.Variable(np.random.normal(loc=0, size=(1, 1, 2)), dtype=tf.float64))*-3
discount_slope_hier = tf.math.sigmoid(tf.Variable(np.random.normal(loc=0, size=(*dim_size, 2)), dtype=tf.float64))*-3

roi_mults_global = tf.Variable(np.random.normal(loc=0, size=(1, 1, me_size)), dtype=tf.float64)
roi_mults_hier = tf.Variable(np.random.normal(loc=0, size=(*dim_size, me_size)), dtype=tf.float64)

nr_to_vol_slope = tf.Variable(np.random.normal(loc=0, size=dim_size), dtype=tf.float64)


hier_var_list = [baseline_slope1_hier, mixed_effect_mult_hier, discount_slope_hier, roi_mults_hier] #baseline_slope2_hier
global_var_list = [baseline_slope1_global, mixed_effect_mult_global, discount_slope_global, roi_mults_global, nr_to_vol_slope] #baseline_slope2_global

discounts_var = tf.Variable(initial_discount_var, dtype=tf.float64)
sliced_discount_var = tf.slice(discounts_var, begin=[0,0,0], size=[2,151,-1])
@tf.function
def model(
        base_intercept_in,
        base_slope1_global_in,
        base_slope1_hier_in,
        # base_slope2_global_in,
        # base_slope2_hier_in,
        mixed_effect_mult_global_in,
        mixed_effect_mult_hier_in,
        discount_slope_global_in,
        discount_slope_hier_in,
        roi_mults_global_in,
        roi_mults_hier_in,
        nr_to_vol_slope_in,
        time_index_in,
        mixed_effect_in,
        discounts_in,
        y_mask_in,
    ):
    base_slope1_in = base_slope1_global_in + base_slope1_hier_in
    # base_slope2_in = base_slope2_global_in + base_slope2_hier_in
    mixed_effect_mult_in = mixed_effect_mult_global_in + mixed_effect_mult_hier_in
    discount_slope_in = discount_slope_global_in + discount_slope_hier_in
    roi_mults_in = roi_mults_global_in + roi_mults_hier_in

    base1_in = tf.multiply(base_slope1_in, time_index_in) + base_intercept_in
    base2_in = base1_in #+ tf.multiply(base_slope2_in, shifted_nr)
    mixed_effect_impact_in = 1 + tf.nn.tanh(tf.multiply(mixed_effect_in, mixed_effect_mult_in))
    total_mixed_effect_impact_in = tf.reduce_prod(mixed_effect_impact_in, axis=-1)
    discount_impact_in = tf.multiply(discount_slope_in, discounts_in)
    roi_mult_impact_in = 1 + tf.nn.tanh(tf.multiply(mixed_effect_impact_in, roi_mults_in))
    total_roi_mult_impact_in = tf.expand_dims(tf.reduce_prod(roi_mult_impact_in, axis=1), axis=1)

    y_pred_out = tf.multiply(
        y_mask_in,
        (
            tf.multiply(base2_in, total_mixed_effect_impact_in)
            + tf.reduce_sum(discount_impact_in, axis=-1)
        )
    )

    y_vol_pred_out = tf.multiply(y_pred_out, nr_to_vol_slope_in)

    return y_pred_out, y_vol_pred_out

@tf.function
def wape(y_actual, y_prediction):
    return tf.reduce_sum(tf.math.abs(y_actual - y_prediction))/tf.reduce_sum(y_actual)

@tf.function
def mse(y_actual, y_prediction):
    return tf.reduce_sum(tf.math.square(y_actual - y_prediction))
y_pred, y_vol_pred = model(
    baseline_intercept,
    baseline_slope1_global,
    baseline_slope1_hier,
    # baseline_slope2_global,
    # baseline_slope2_hier_in,
    mixed_effect_mult_global,
    mixed_effect_mult_hier,
    discount_slope_global,
    discount_slope_hier,
    roi_mults_global,
    roi_mults_hier,
    nr_to_vol_slope,
    time_index,
    mixed_effect,
    discounts,
    y_mask,
)

y_split = tf.split(y, val_splitter)
y_pred_split = tf.split(y_pred, val_splitter)

y_vol_split = tf.split(y_vol, val_splitter)
y_vol_pred_split = tf.split(y_vol_pred, val_splitter)


# loss
total_wape = tf.math.reduce_mean([wape(y_split[i], y_pred_split[i]) for i in range(0,val_splitter)])
total_mse = mse(y, y_pred)
actual_wape = wape(y, y_pred)

total_wape_vol = tf.math.reduce_mean([wape(y_vol_split[i], y_vol_pred_split[i]) for i in range(0,val_splitter)])
total_mse_vol = mse(y_vol, y_vol_pred)
actual_wape_vol = wape(y_vol, y_vol_pred)


reg1 = sum([tf.reduce_sum(tf.square(i)) for i in hier_var_list])
reg2 = sum([tf.reduce_sum(tf.square(i)) for i in global_var_list])

loss = (
    1e3*total_wape_vol
    +1e1*total_mse_vol
    +1e3*total_wape
    +1e1*total_mse
    +1e3*reg2
    +1e1*reg1
)
@tf.function
def _tensor_gather(tensor_to_gather_in, encoding, key):
    encoding = pd.Series(encoding["mapper_dict"][key]).sort_index().to_numpy()
    segment_ids = tf.constant(encoding, dtype=tf.int32)
    x_transpose = tf.transpose(tensor_to_gather_in, perm=[1,0,2])
    x_gathered = tf.math.unsorted_segment_sum(x_transpose, segment_ids, num_segments=encoding.max()+1)
    x_gathered_transpose = tf.reduce_mean(tf.transpose(x_gathered, perm=[1,0,2]), axis=2)

    return x_gathered_transpose
y_inital_pred, y_initial_vol_pred = model(
    baseline_intercept,
    baseline_slope1_global,
    baseline_slope1_hier,
    # baseline_slope2_global,
    # baseline_slope2_hier_in,
    mixed_effect_mult_global,
    mixed_effect_mult_hier,
    discount_slope_global,
    discount_slope_hier,
    roi_mults_global,
    roi_mults_hier,
    nr_to_vol_slope,
    time_index_var,
    mixed_effect_var,
    initial_discount_var,
    y_mask_var,
)

slice_y_inital_pred = tf.slice(y_inital_pred, begin=[0,0], size=[-1, 151])
slice_y_initial_vol_pred = tf.slice(y_initial_vol_pred, begin=[0,0], size=[-1, 151])


y_opt_pred, y_opt_vol_pred = model(
    baseline_intercept,
    baseline_slope1_global,
    baseline_slope1_hier,
    # baseline_slope2_global,
    # baseline_slope2_hier_in,
    mixed_effect_mult_global,
    mixed_effect_mult_hier,
    discount_slope_global,
    discount_slope_hier,
    roi_mults_global,
    roi_mults_hier,
    nr_to_vol_slope,
    time_index_var,
    mixed_effect_var,
    discounts_var,
    y_mask_var,
)

slice_y_opt_pred = tf.slice(y_opt_pred, begin=[0,0], size=[-1, 151])
slice_y_opt_vol_pred = tf.slice(y_opt_vol_pred, begin=[0,0], size=[-1, 151])


discount_var_brand = _tensor_gather(sliced_discount_var, final_encodings, "brand")
discount_var_pack = _tensor_gather(sliced_discount_var, final_encodings, "pack")
discount_var_segment = _tensor_gather(sliced_discount_var, final_encodings, "segment")

brand_constraint_loss = tf.reduce_sum(tf.nn.relu(brand_constraint_tensor - discount_var_brand))
pack_constraint_loss = tf.reduce_sum(tf.nn.relu(pack_constraint_tensor - discount_var_pack))
segment_constraint_loss = tf.reduce_sum(tf.nn.relu(segment_constraint_tensor - discount_var_segment))

roi = tf.divide(tf.reduce_sum(slice_y_opt_pred - slice_y_inital_pred), -tf.reduce_sum(discounts_var))

loss_roi = -1e2*roi + 1e1*brand_constraint_loss + 1e1*pack_constraint_loss + 1e1*segment_constraint_loss
splitter = 40

feed_dict1 = {
    discounts : discount_data[:splitter],
    mixed_effect: macro_data[:splitter],
    y_vol : volume_data[:splitter],
    y : nr_data[:splitter],
    # shifted_nr : nr_shifted[:splitter],
    y_mask : nr_data_mask[:splitter],
    time_index : time_index_array[:splitter],
    val_splitter_ : 5,
    initial_discount_var : discount_data[-2:],
    mixed_effect_var : macro_data[-2:],
    time_index_var : time_index_array[-2:],
    y_mask_var : nr_data_mask[-2:]
}

feed_dict2 = {
    discounts : discount_data[splitter:-5],
    mixed_effect: macro_data[splitter:-5],
    y_vol : volume_data[splitter:-5],
    y : nr_data[splitter:-5],
    # shifted_nr : nr_shifted[splitter:-5],
    y_mask : nr_data_mask[splitter:-5],
    time_index : time_index_array[splitter:-5],
    val_splitter_ : 5,
    initial_discount_var : discount_data[-2:],
    mixed_effect_var : macro_data[-2:],
    time_index_var : time_index_array[-2:],
    y_mask_var : nr_data_mask[-2:]
}

# feed_dict3 = {
#     discounts : discount_data[-2:],
#     mixed_effect: macro_data[-2:],
#     y_vol : volume_data[-2:],
#     y : nr_data[-2:],
#     # shifted_nr : nr_shifted[-2:],
#     y_mask : nr_data_mask[-2:],
#     time_index : time_index_array[-2:],
#     val_splitter_ : 5
# }


initial_discount_var = tf.compat.v1.placeholder(dtype=tf.float64, name="initial_discount_submit")

mixed_effect_var = tf.compat.v1.placeholder(dtype=tf.float64, name="mixed_effect_submit")
time_index_var = tf.compat.v1.placeholder(dtype=tf.float64, name="time_index_submit")
y_mask_var = tf.compat.v1.placeholder(dtype=tf.float64, name="y_mask_submit")
epoch = 0
# optimizer
lr = lambda x : 1 / np.power(x/5 + 10, 1/2)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr(epoch))#, beta1=0.1, beta2=0.1)
train = optimizer.minimize(loss)
[lr(i) for i in [0, 1, 10, 100, 1000, 10000, 20000]]#, 50000, 80000]]
[0.31622776601683794,
 0.31311214554257477,
 0.2886751345948129,
 0.18257418583505536,
 0.06900655593423542,
 0.02230498683727353,
 0.015791661046371634]
# epoch = 0
# # optimizer
# lr2 = lambda x : 1 / np.power(x/5 + 10, 1/2)
# optimizer_roi = tf.compat.v1.train.AdamOptimizer(learning_rate=lr2(epoch))#, beta1=0.1, beta2=0.1)
# train_roi = optimizer_roi.minimize(loss_roi, var_list=[discounts_var])
# initialize variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init, feed_dict1)
2023-09-10 19:51:42.205554: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
metric_update_track = {
    "epoch" : [],
    "actual_wape" : [],
    "test_wape" : [],
    "loss" : [],
    "mse" : [],
    "reg1" : [],
    "reg2" : []
}

# train model
num_epochs = 2000
for epoch in range(num_epochs):
    (
        _,
        current_loss,
        current_wape,
        # current_mse,
        current_wape_vol,
        # current_mse_vol,
        current_reg1,
        current_reg2
    )= sess.run([
        train,
        loss,
        actual_wape,
        # total_mse,
        actual_wape_vol,
        # total_mse_vol,
        reg1,
        reg2
    ], feed_dict1)

    current_wape_test, current_wape_vol_test = sess.run([actual_wape, actual_wape_vol], feed_dict2)


    if (epoch + 1) % 250 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {current_loss:.4f}, WAPE: {current_wape:.4f}, WAPE_TEST: {current_wape_test:.4f}, WAPE_VOL: {current_wape_vol:.4f}, WAPE_VOL_TEST: {current_wape_vol_test:.4f}, reg1: {current_reg1:.4f}, reg2: {current_reg2:.4f}")
        # metric_update_track["epoch"].append(epoch)
        # metric_update_track["actual_wape"].append(current_wape)
        # metric_update_track["test_wape"].append(current_wape_test)
        # metric_update_track["loss"].append(current_loss)
        # metric_update_track["mse"].append(current_mse)
        # metric_update_track["reg1"].append(current_reg1)
        # metric_update_track["reg2"].append(current_reg2)



#         # Training loop
# num_epochs = 500
# for epoch in range(num_epochs):
#     _, current_error, cuurent_mse, current_m1, current_m2, current_c = sess.run([train_op, error, mse_error, m1, m2, c])
#     if (epoch + 1) % 25 == 0:
#         print(f"Epoch {epoch + 1}/{num_epochs}, Error: {current_error:.4f}, MSE: {cuurent_mse:.4f}, m1: {current_m1}, m2: {current_m2}, c: {current_c}")

# # Print the final results for 'm' and 'c'
# final_m1, final_m2, final_c = sess.run([m1, m2, c])
# print(f"Final 'm1' value: {final_m1}")

# print(f"Final 'm2' value: {final_m2}")
# print(f"Final 'c' value: {final_c}")

output = sess.run([y_inital_pred, y_initial_vol_pred], feed_dict1)
def convert_forcast_format_to_optimised(fd):
    output = sess.run([y_inital_pred, y_initial_vol_pred], fd)
    output_df = pd.DataFrame(output[1],index=[6,7],columns=total_sku_list).stack(0).reset_index()
    output_df["sku"] = output_df.apply(lambda x: x["level_1"]+"_"+str(x["level_0"]),axis=1)
    output_df = output_df[output_df.sku.isin(sales_data_opt.apply(lambda x: x["sku_for"]+"_"+str(x["month_for"]),axis=1))].drop(["level_1"],axis=1)
    output_df[output_df.sku.isin(sales_data_opt["sku_month"])]
    forecast_volumes = output_df.groupby(["level_0","sku"])[0].sum().unstack(1)[sales_data_opt["sku_month"]].sum().values
    return forecast_volumes
budgeted_forecast_volumes = convert_forcast_format_to_optimised(feed_dict1)

# def convert_promo_budget_to_forcast_format():
#     arr1 = sales_data_opt["Promotional_Discount_LC"].values
#     promo1_df = pd.DataFrame(index=[6,7],columns=total_sku_list)
#     promo1_df.loc[6, sales_data_opt.set_index(["month_for", "sku_for"])["Promotional_Discount_LC"].unstack(1).columns] = arr1[::2]
#     promo1_df.loc[7, sales_data_opt.set_index(["month_for", "sku_for"])["Promotional_Discount_LC"].unstack(1).columns] = arr1[1::2]
#     promo1_df = promo1_df.fillna(0.0)

#     arr2 = sales_data_opt["Other_Discounts_LC"].values
#     promo2_df = pd.DataFrame(index=[6,7],columns=total_sku_list)
#     promo2_df.loc[6, sales_data_opt.set_index(["month_for", "sku_for"])["Other_Discounts_LC"].unstack(1).columns] = arr1[::2]
#     promo2_df.loc[7, sales_data_opt.set_index(["month_for", "sku_for"])["Other_Discounts_LC"].unstack(1).columns] = arr1[1::2]
#     promo2_df = promo2_df.fillna(0.0)

#     return np.stack((promo1_df.values, promo2_df.values),axis=2)
# convert_promo_budget_to_forcast_format().shape

def convert_promo_budget_to_forcast_format(promo1_investments,promo2_investments):
    arr1 = promo1_investments
    promo1_df = pd.DataFrame(index=[6,7],columns=total_sku_list)
    promo1_df.loc[6, sales_data_opt.set_index(["month_for", "sku_for"])["Promotional_Discount_LC"].unstack(1).columns] = arr1[::2]
    promo1_df.loc[7, sales_data_opt.set_index(["month_for", "sku_for"])["Promotional_Discount_LC"].unstack(1).columns] = arr1[1::2]
    promo1_df = promo1_df.fillna(0.0)
    
    arr2 = promo2_investments
    promo2_df = pd.DataFrame(index=[6,7],columns=total_sku_list)
    promo2_df.loc[6, sales_data_opt.set_index(["month_for", "sku_for"])["Other_Discounts_LC"].unstack(1).columns] = arr1[::2]
    promo2_df.loc[7, sales_data_opt.set_index(["month_for", "sku_for"])["Other_Discounts_LC"].unstack(1).columns] = arr1[1::2]
    promo2_df = promo2_df.fillna(0.0)
    
    return np.stack((promo1_df.values, promo2_df.values),axis=2)

# Required Libraries
import pandas as pd
import numpy
from deap import base, creator, tools, algorithms
import random
# Load Data
sales_data = pd.read_excel("sales_data_hackathon.xlsx")
macro_data = pd.read_excel("macro_data.xlsx")
brand_segment_mapping = pd.read_excel("brand_segment_mapping_hackathon.xlsx")
volume_constraints = pd.read_excel("volume_variation_constraint_hackathon.xlsx")
pack_constraints = pd.read_excel("maximum_discount_constraint_hackathon.xlsx", sheet_name="Pack")
brand_constraints = pd.read_excel("maximum_discount_constraint_hackathon.xlsx")
price_segment_constraints = pd.read_excel("maximum_discount_constraint_hackathon.xlsx", sheet_name="PriceSegment")
sample_output = pd.read_csv("submission_template_hackathon.csv")

sales_data["SBPS"] = sales_data.apply(lambda x: x["SKU"]+"_"+x["Brand"]+"_"+x["Pack"]+"_"+x["Size"]+"_"+str(x["Year"])+"_"+str(x["Month"]),axis=1)
brand_segment_mapping_dict = dict(zip(brand_segment_mapping.Brand, brand_segment_mapping.PriceSegment))
sales_data["segment"] = sales_data.Brand
sales_data = sales_data.replace({"segment":brand_segment_mapping_dict })
sales_data = sales_data.drop(columns=["SKU", "Brand", "Pack", "Size", "MACO_LC", "VILC_LC", "Total_Discounts_LC", "Year","Month"])
volume_constraints["SBPS"] = volume_constraints.apply(lambda x: x["SKU"]+"_"+x["Brand"]+"_"+x["Pack"]+"_"+x["Size"],axis=1)
volume_constraints = volume_constraints.drop(columns=["SKU", "Brand", "Pack", "Size"])

sample_output["SBPS"] = sample_output.apply(lambda x: x["SKU"]+"_"+x["Brand"]+"_"+x["Pack"]+"_"+x["Size"]+"_"+str(x["Year"])+"_"+str(x["Month"]),axis=1)
sample_output = sample_output.drop(columns=["SKU", "Brand", "Pack", "Size","Volume_Htls", "Net_Revenue_LC", "Year","Month"])
ouput_sbps_vals = sample_output.SBPS
ouput_sbps_without_time = ouput_sbps_vals.str.rsplit("_",n=2,expand=True)[0]
sales_data_opt = sales_data[sales_data.SBPS.isin(ouput_sbps_vals)]
#fake data
sales_data_opt["Volume_Htls"] = sales_data.Volume_Htls.iloc[:194].values
sales_data_opt = sales_data_opt.reset_index(drop=True)
sales_data_opt[["month_for", "sku_for"]] = sales_data_opt.SBPS.str.split("_", expand=True)[[5,0]]
sales_data_opt["sku_month"] = sales_data_opt.apply(lambda x: x["sku_for"]+"_"+str(x["month_for"]),axis=1)
sbps_not_in_volume = list(set(ouput_sbps_without_time) - set(volume_constraints.SBPS))
volume_constraints = pd.concat([volume_constraints,pd.DataFrame(sbps_not_in_volume, columns=["SBPS"])]).fillna(0.0)
from sklearn.preprocessing import MinMaxScaler
# Scaling data
scaler_features = MinMaxScaler()
# features_scaled = scaler_features.fit_transform(sales_data_opt[["Promotional_Discount_LC", "Other_Discounts_LC","Volume_Htls"]])

sales_data_opt.loc[:,["Promotional_Discount_LC", "Other_Discounts_LC","Volume_Htls"]] = scaler_features.fit_transform(sales_data_opt[["Promotional_Discount_LC", "Other_Discounts_LC","Volume_Htls"]])

pack_constraints.loc[:,"max_discount"] = scaler_features.fit_transform(pack_constraints["max_discount"].values.reshape(-1, 1))
brand_constraints.loc[:,"max_discount"] = scaler_features.fit_transform(brand_constraints["max_discount"].values.reshape(-1, 1))
price_segment_constraints.loc[:,"max_discount"] = scaler_features.fit_transform(price_segment_constraints["max_discount"].values.reshape(-1, 1))
# Define the problem as a maximization
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))  # Maximize adjusted volume and minimize investment
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Initialization for promo investments
# Here we're initializing the investments for each SKU in two promotion categories
def get_promo_data(promo_name):
    return sales_data_opt[promo_name].values


toolbox.register("promo1", get_promo_data, "Promotional_Discount_LC")
toolbox.register("promo2", get_promo_data, "Other_Discounts_LC")
# toolbox.register("promo1", random.uniform, 0, 1000)
# toolbox.register("promo2", random.uniform, 0, 1000)

# An individual represents investments for all SKUs across both promotions
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.promo1, toolbox.promo2),
    n=1,
    # n=len(sales_data_opt["SBPS"].unique()),
)

# A population will be a collection of these individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Skus from volume constrints not in sales data
vol_skus_not_in_sales = list(
    set(volume_constraints.SBPS.unique()) - set((sales_data_opt.SBPS.str.rsplit("_", n=2, expand=True)[0]).unique())
)

# function to get sku index
def get_sku_index(sku_name, volume_sku=False):
    if volume_sku:
        idx = sales_data_opt[sales_data_opt.SBPS.str.rsplit("_", n=2, expand=True)[0] == sku_name].index.values
    else:
        idx = sales_data_opt[sales_data_opt.SBPS == sku_name].index.values
    return idx


# Evaluation function
def evaluate(individual):
    # Split the investments
    # promo1_investments = individual[::2]
    # promo2_investments = individual[1::2]
    promo1_investments = individual[0]
    promo2_investments = individual[1]

    # Basic volume model based on the investments
    splitter = 40
    feed_dict1 = {
        discounts : discount_data[:splitter],
        mixed_effect: macro_data[:splitter],
        y_vol : volume_data[:splitter],
        y : nr_data[:splitter],
        # shifted_nr : nr_shifted[:splitter],
        y_mask : nr_data_mask[:splitter],
        time_index : time_index_array[:splitter],
        val_splitter_ : 5,
        initial_discount_var : convert_promo_budget_to_forcast_format(promo1_investments,promo2_investments),
        mixed_effect_var : macro_data[-2:],
        time_index_var : time_index_array[-2:],
        y_mask_var : nr_data_mask[-2:]
    }
    optimised_budget_vol_forecast = sess.run([y_inital_pred, y_initial_vol_pred], feed_dict1)
    volume_increase = budgeted_forecast_volumes - optimised_budget_vol_forecast
    total_investment = sum(promo1_investments) + sum(promo2_investments)

    # Initialize penalty for constraint violations
    constraint_penalty = 0

    # Constraint 1: Volume variation
    for _, row in volume_constraints.iterrows():
        if row["SBPS"] in vol_skus_not_in_sales:
            continue
        else:
            sku = get_sku_index(
                row["SBPS"], volume_sku=True
            )  # lenght of sku will be 2 for volume. one for 6th and one for 7th month
        min_variation = row["Minimum Volume Variation"]
        max_variation = row["Maximum Volume Variation"]

        estimated_volume_variation = promo1_investments[sku] * 0.05 + promo2_investments[sku] * 0.07
        estimated_volume_variation_monthly = (estimated_volume_variation[1] - estimated_volume_variation[0])/estimated_volume_variation[0]
        if estimated_volume_variation_monthly < min_variation or estimated_volume_variation_monthly > max_variation:
            constraint_penalty += 1000

    # Constraint 2: Pack discount
    for _, row in pack_constraints.iterrows():
        pack, month, year = row[["Pack", "Month", "Year"]].values
        max_discount_per_pack = row["max_discount"]

        skus_in_pack = sales_data_opt[
            (sales_data_opt["SBPS"].str.split("_", expand=True)[2] == pack)
            & (sales_data_opt["SBPS"].str.split("_", expand=True)[4] == str(year))
            & (sales_data_opt["SBPS"].str.split("_", expand=True)[5] == str(month))
        ]["SBPS"].unique()
        avg_discount = sum(
            promo1_investments[get_sku_index(sku)] + promo2_investments[get_sku_index(sku)] for sku in skus_in_pack
        ) / len(skus_in_pack)

        if avg_discount > max_discount_per_pack:
            constraint_penalty += 1000

    # Constraint 3: Brand discount
    for _, row in brand_constraints.iterrows():
        brand, month, year = row[["Brand", "Month", "Year"]].values
        max_discount_per_brand = row["max_discount"]
        skus_in_brand = sales_data_opt[
            (sales_data_opt["SBPS"].str.split("_", expand=True)[1] == brand)
            & (sales_data_opt["SBPS"].str.split("_", expand=True)[4] == str(year))
            & (sales_data_opt["SBPS"].str.split("_", expand=True)[5] == str(month))
        ]["SBPS"].unique()
        if len(skus_in_brand) != 0:
            avg_discount_brand = sum(
                promo1_investments[get_sku_index(sku)] + promo2_investments[get_sku_index(sku)] for sku in skus_in_brand
            ) / len(skus_in_brand)
            if avg_discount_brand > max_discount_per_brand:
                constraint_penalty += 1000

    # Constraint 4: PriceSegment discount
    for _, row in price_segment_constraints.iterrows():
        segment, month, year = row[["PriceSegment", "Month", "Year"]].values
        max_discount_price_segment = row["max_discount"]
        skus_in_segment = sales_data_opt[
            (sales_data_opt["segment"] == segment)
            & (sales_data_opt["SBPS"].str.split("_", expand=True)[4] == str(year))
            & (sales_data_opt["SBPS"].str.split("_", expand=True)[5] == str(month))
        ]["SBPS"].unique()
        avg_discount_segment = sum(
            promo1_investments[get_sku_index(sku)] + promo2_investments[get_sku_index(sku)] for sku in skus_in_segment
        ) / len(skus_in_segment)

        if avg_discount_segment > max_discount_price_segment:
            constraint_penalty += 1000
    # Constraint 5: Net volume for each SKU should be positive
    for sku in sales_data_opt["SBPS"].unique():
        net_volume_sku = (
            sales_data_opt[sales_data_opt["SBPS"] == sku]["Volume_Htls"].sum()
            + promo1_investments[get_sku_index(sku)] * 0.1
            + promo2_investments[get_sku_index(sku)] * 0.15
        )
        if net_volume_sku < 0:
            constraint_penalty += 5000  # Assigning a significant penalty for negative net volume
    # Constraint 6: Maximum increase in next month’s total optimal volume at an overall level should not exceed 9.1%
    net_optimal_volume = sales_data_opt["Volume_Htls"] + promo1_investments * 0.1 + promo2_investments * 0.15
    net_monthly_uplift = net_optimal_volume[1::2].sum() - net_optimal_volume[::2].sum()
    percent_increment = net_monthly_uplift / (net_optimal_volume[::2].sum())
    if percent_increment > 1.091:
        constraint_penalty += 5000
    # Adjust the volume based on penalties
    adjusted_volume = volume_increase - constraint_penalty

    return adjusted_volume, total_investment


# Define the genetic operators
toolbox.register("mate", tools.cxTwoPoint)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.1)  # Mutation
toolbox.register("select", tools.selNSGA2)  # Selection
toolbox.register("evaluate", evaluate)  # Evaluation


# Initialize population and run the GA loop
population = toolbox.population(n=300)
NGEN = 1
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)  # Apply crossover and mutation
    fits = toolbox.map(toolbox.evaluate, offspring)  # Evaluate the offspring
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit  # Assign fitness
    population = toolbox.select(offspring, k=len(population))  # Select the next generation individuals


# Print the best solution
top1 = tools.selBest(population, k=1)[0]
print(top1)
print(toolbox.evaluate(top1))
stop
nr_submit, vol_submit = sess.run([y_pred, y_vol_pred], feed_dict3)
nr_submit = nr_submit * scaler
vol_submit = vol_submit * vol_scaler

nr_data_temp = (
    sales_data
    .reset_index()
    .groupby(["date", "sku"])
    .net_revenue.sum()
    .sort_index()
    .unstack(1)
)
nr_submit = pd.DataFrame(nr_submit, index=nr_data_temp.index[-2:], columns=nr_data_temp.columns)
vol_submit = pd.DataFrame(vol_submit, index=nr_data_temp.index[-2:], columns=nr_data_temp.columns)

submit_temp = sales_data[sales_data.gto.isna()].reset_index().set_index(["date", "sku", "brand", "pack", "size"]).sort_index()
submit_temp.loc[:, "net_revenue"] = submit_temp.net_revenue.fillna(nr_submit.stack()).apply(lambda x: x if x>0 else -x/2)
submit_temp.loc[:, "volume"] = submit_temp.volume.fillna(vol_submit.stack()).apply(lambda x: x if x>0 else -x/2)
submit_temp = submit_temp.reset_index()

cols_req = [ "Year", "Month", "SKU", "Brand", "Pack", "Size", "Volume_Estimate", "Net_Revenue_Estimate", "Optimal_Promotional_Discount", "Optimal_Other_Discounts", "Optimal_Volume", "Optimal_Net_Revenue"]


submit_temp.loc[:, "Year"] = submit_temp.date.dt.year
submit_temp.loc[:, "Month"] = submit_temp.date.dt.month
submit_temp.loc[:, "SKU"] = submit_temp.sku
submit_temp.loc[:, "Brand"] = submit_temp.brand
submit_temp.loc[:, "Pack"] = submit_temp.pack
submit_temp.loc[:, "Size"] = submit_temp.size
submit_temp.loc[:, "Volume_Estimate"] = submit_temp.volume
submit_temp.loc[:, "Net_Revenue_Estimate"] = submit_temp.net_revenue
submit_temp.loc[:, "Optimal_Promotional_Discount"] = submit_temp.promotional_discount
submit_temp.loc[:, "Optimal_Other_Discounts"] = submit_temp.other_discounts
submit_temp.loc[:, "Optimal_Volume"] = submit_temp.volume
submit_temp.loc[:, "Optimal_Net_Revenue"] = submit_temp.net_revenue

submit_temp = submit_temp[cols_req]