import streamlit as st
import pandas as pd
import joblib
import numpy as np
import category_encoders
import sklearn
import xgboost


PROPERTY_TYPES = ['Independent House', 'Apartment', 'Villa']

FURNISHED_STATUS = ['Semi-furnished', 'Furnished', 'Unfurnished']

OWNER_TYPES = ['Builder', 'Owner', 'Broker']

ORDINAL_COLS = {
    "Public_Transport_Accessibility": ["Low", "Medium", "High"],
    "Facing": ["East", "West", "North", "South"]
}

BINARY_COLS = ["Parking_Space", "Security", "Availability_Status"]

STATES = [
    'Haryana','Andhra Pradesh','Madhya Pradesh','Punjab','Delhi',
    'Maharashtra','Karnataka','Jharkhand','Tamil Nadu','Chhattisgarh',
    'Kerala','Odisha','West Bengal','Bihar','Rajasthan','Uttar Pradesh',
    'Telangana','Assam','Uttarakhand','Gujarat'
]

CITIES = [
    'Gurgaon','Vishakhapatnam','Bhopal','Ludhiana','Faridabad','New Delhi',
    'Indore','Pune','Bangalore','Jamshedpur','Chennai','Mangalore',
    'Raipur','Trivandrum','Coimbatore','Bilaspur','Bhubaneswar',
    'Durgapur','Vijayawada','Gaya','Mumbai','Amritsar','Jodhpur',
    'Lucknow','Hyderabad','Guwahati','Silchar','Haridwar','Ranchi',
    'Kolkata','Nagpur','Dehradun','Noida','Patna','Cuttack','Warangal',
    'Ahmedabad','Jaipur','Kochi','Dwarka','Mysore','Surat'
]

LOCALITIES = ['Locality_123', 'Locality_74', 'Locality_486', 'Locality_13',
       'Locality_22', 'Locality_62', 'Locality_248', 'Locality_464',
       'Locality_215', 'Locality_127', 'Locality_346', 'Locality_283',
       'Locality_226', 'Locality_231', 'Locality_14', 'Locality_38',
       'Locality_311', 'Locality_409', 'Locality_83', 'Locality_108',
       'Locality_385', 'Locality_106', 'Locality_351', 'Locality_292',
       'Locality_411', 'Locality_57', 'Locality_284', 'Locality_145',
       'Locality_75', 'Locality_359', 'Locality_261', 'Locality_230',
       'Locality_373', 'Locality_403', 'Locality_198', 'Locality_444',
       'Locality_84', 'Locality_488', 'Locality_433', 'Locality_499',
       'Locality_401', 'Locality_457', 'Locality_32', 'Locality_343',
       'Locality_92', 'Locality_435', 'Locality_295', 'Locality_353',
       'Locality_495', 'Locality_26', 'Locality_59', 'Locality_487',
       'Locality_263', 'Locality_468', 'Locality_442', 'Locality_303',
       'Locality_232', 'Locality_79', 'Locality_246', 'Locality_331',
       'Locality_243', 'Locality_349', 'Locality_150', 'Locality_122',
       'Locality_12', 'Locality_443', 'Locality_479', 'Locality_365',
       'Locality_46', 'Locality_185', 'Locality_313', 'Locality_40',
       'Locality_153', 'Locality_424', 'Locality_63', 'Locality_426',
       'Locality_437', 'Locality_206', 'Locality_105', 'Locality_301',
       'Locality_190', 'Locality_24', 'Locality_478', 'Locality_118',
       'Locality_491', 'Locality_89', 'Locality_197', 'Locality_395',
       'Locality_34', 'Locality_361', 'Locality_186', 'Locality_449',
       'Locality_358', 'Locality_372', 'Locality_470', 'Locality_137',
       'Locality_494', 'Locality_414', 'Locality_6', 'Locality_139',
       'Locality_82', 'Locality_277', 'Locality_49', 'Locality_5',
       'Locality_420', 'Locality_77', 'Locality_390', 'Locality_4',
       'Locality_250', 'Locality_87', 'Locality_233', 'Locality_378',
       'Locality_41', 'Locality_327', 'Locality_133', 'Locality_126',
       'Locality_170', 'Locality_391', 'Locality_196', 'Locality_125',
       'Locality_406', 'Locality_392', 'Locality_431', 'Locality_306',
       'Locality_298', 'Locality_10', 'Locality_356', 'Locality_44',
       'Locality_285', 'Locality_316', 'Locality_308', 'Locality_312',
       'Locality_455', 'Locality_36', 'Locality_389', 'Locality_355',
       'Locality_325', 'Locality_142', 'Locality_147', 'Locality_235',
       'Locality_72', 'Locality_440', 'Locality_287', 'Locality_143',
       'Locality_179', 'Locality_450', 'Locality_329', 'Locality_128',
       'Locality_111', 'Locality_332', 'Locality_47', 'Locality_107',
       'Locality_462', 'Locality_100', 'Locality_493', 'Locality_228',
       'Locality_174', 'Locality_148', 'Locality_208', 'Locality_398',
       'Locality_293', 'Locality_86', 'Locality_104', 'Locality_314',
       'Locality_266', 'Locality_396', 'Locality_161', 'Locality_65',
       'Locality_484', 'Locality_393', 'Locality_17', 'Locality_183',
       'Locality_254', 'Locality_460', 'Locality_429', 'Locality_453',
       'Locality_445', 'Locality_456', 'Locality_7', 'Locality_176',
       'Locality_220', 'Locality_461', 'Locality_204', 'Locality_467',
       'Locality_218', 'Locality_53', 'Locality_251', 'Locality_222',
       'Locality_352', 'Locality_333', 'Locality_114', 'Locality_258',
       'Locality_317', 'Locality_454', 'Locality_402', 'Locality_23',
       'Locality_210', 'Locality_68', 'Locality_290', 'Locality_71',
       'Locality_422', 'Locality_112', 'Locality_483', 'Locality_302',
       'Locality_181', 'Locality_280', 'Locality_366', 'Locality_151',
       'Locality_368', 'Locality_310', 'Locality_296', 'Locality_239',
       'Locality_93', 'Locality_76', 'Locality_340', 'Locality_459',
       'Locality_337', 'Locality_273', 'Locality_432', 'Locality_39',
       'Locality_234', 'Locality_163', 'Locality_386', 'Locality_262',
       'Locality_101', 'Locality_326', 'Locality_78', 'Locality_447',
       'Locality_374', 'Locality_405', 'Locality_279', 'Locality_275',
       'Locality_212', 'Locality_436', 'Locality_20', 'Locality_152',
       'Locality_500', 'Locality_61', 'Locality_256', 'Locality_375',
       'Locality_189', 'Locality_131', 'Locality_466', 'Locality_113',
       'Locality_33', 'Locality_481', 'Locality_67', 'Locality_371',
       'Locality_80', 'Locality_29', 'Locality_253', 'Locality_241',
       'Locality_376', 'Locality_194', 'Locality_136', 'Locality_267',
       'Locality_66', 'Locality_240', 'Locality_132', 'Locality_370',
       'Locality_115', 'Locality_64', 'Locality_397', 'Locality_103',
       'Locality_322', 'Locality_18', 'Locality_271', 'Locality_347',
       'Locality_335', 'Locality_247', 'Locality_130', 'Locality_91',
       'Locality_155', 'Locality_419', 'Locality_214', 'Locality_16',
       'Locality_439', 'Locality_211', 'Locality_474', 'Locality_223',
       'Locality_400', 'Locality_307', 'Locality_37', 'Locality_164',
       'Locality_157', 'Locality_140', 'Locality_430', 'Locality_489',
       'Locality_252', 'Locality_180', 'Locality_330', 'Locality_476',
       'Locality_98', 'Locality_281', 'Locality_172', 'Locality_336',
       'Locality_158', 'Locality_404', 'Locality_289', 'Locality_288',
       'Locality_354', 'Locality_229', 'Locality_269', 'Locality_159',
       'Locality_417', 'Locality_99', 'Locality_291', 'Locality_427',
       'Locality_328', 'Locality_469', 'Locality_160', 'Locality_242',
       'Locality_300', 'Locality_28', 'Locality_255', 'Locality_475',
       'Locality_69', 'Locality_496', 'Locality_425', 'Locality_490',
       'Locality_15', 'Locality_187', 'Locality_166', 'Locality_90',
       'Locality_149', 'Locality_268', 'Locality_19', 'Locality_45',
       'Locality_31', 'Locality_299', 'Locality_345', 'Locality_416',
       'Locality_94', 'Locality_207', 'Locality_408', 'Locality_344',
       'Locality_121', 'Locality_144', 'Locality_225', 'Locality_472',
       'Locality_278', 'Locality_412', 'Locality_452', 'Locality_260',
       'Locality_165', 'Locality_192', 'Locality_318', 'Locality_297',
       'Locality_58', 'Locality_167', 'Locality_85', 'Locality_48',
       'Locality_209', 'Locality_264', 'Locality_294', 'Locality_304',
       'Locality_413', 'Locality_227', 'Locality_73', 'Locality_120',
       'Locality_124', 'Locality_188', 'Locality_384', 'Locality_383',
       'Locality_458', 'Locality_110', 'Locality_88', 'Locality_339',
       'Locality_338', 'Locality_341', 'Locality_448', 'Locality_438',
       'Locality_102', 'Locality_138', 'Locality_244', 'Locality_95',
       'Locality_360', 'Locality_21', 'Locality_70', 'Locality_178',
       'Locality_199', 'Locality_169', 'Locality_272', 'Locality_377',
       'Locality_410', 'Locality_379', 'Locality_446', 'Locality_25',
       'Locality_191', 'Locality_81', 'Locality_193', 'Locality_2',
       'Locality_117', 'Locality_56', 'Locality_216', 'Locality_423',
       'Locality_270', 'Locality_54', 'Locality_109', 'Locality_421',
       'Locality_324', 'Locality_257', 'Locality_51', 'Locality_184',
       'Locality_415', 'Locality_274', 'Locality_323', 'Locality_485',
       'Locality_205', 'Locality_27', 'Locality_116', 'Locality_30',
       'Locality_388', 'Locality_480', 'Locality_168', 'Locality_382',
       'Locality_319', 'Locality_276', 'Locality_162', 'Locality_171',
       'Locality_362', 'Locality_8', 'Locality_399', 'Locality_463',
       'Locality_97', 'Locality_146', 'Locality_394', 'Locality_471',
       'Locality_182', 'Locality_11', 'Locality_357', 'Locality_367',
       'Locality_434', 'Locality_387', 'Locality_201', 'Locality_305',
       'Locality_236', 'Locality_369', 'Locality_119', 'Locality_217',
       'Locality_428', 'Locality_50', 'Locality_477', 'Locality_482',
       'Locality_321', 'Locality_348', 'Locality_334', 'Locality_175',
       'Locality_224', 'Locality_213', 'Locality_249', 'Locality_203',
       'Locality_156', 'Locality_465', 'Locality_315', 'Locality_221',
       'Locality_418', 'Locality_492', 'Locality_320', 'Locality_342',
       'Locality_238', 'Locality_177', 'Locality_441', 'Locality_309',
       'Locality_364', 'Locality_52', 'Locality_9', 'Locality_286',
       'Locality_202', 'Locality_60', 'Locality_350', 'Locality_134',
       'Locality_473', 'Locality_219', 'Locality_451', 'Locality_1',
       'Locality_141', 'Locality_42', 'Locality_96', 'Locality_154',
       'Locality_497', 'Locality_135', 'Locality_245', 'Locality_380',
       'Locality_407', 'Locality_363', 'Locality_498', 'Locality_265',
       'Locality_282', 'Locality_200', 'Locality_35', 'Locality_3',
       'Locality_129', 'Locality_381', 'Locality_173', 'Locality_43',
       'Locality_237', 'Locality_259', 'Locality_195', 'Locality_55']
        # full Locality array (unchanged)


st.set_page_config(page_title="Property Investment Intelligence", layout="wide")
st.title("🏠 Property Investment Intelligence Platform")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("GradientBoostingClassifier_model_pipeline.joblib")
    reg = joblib.load("XGB_model_pipeline.joblib")
    return clf, reg

clf_model, reg_model = load_models()

# -----------------------------
# Mode Selection
# -----------------------------
mode = st.sidebar.radio(
    "Prediction Type",
    ["Investment Classification", "Future Price Prediction"]
)

# -----------------------------
# Basic Property Info
# -----------------------------
st.subheader("📌 Property Details")
c1, c2, c3 = st.columns(3)

with c1:
    State = st.selectbox("State", STATES)
    City = st.selectbox("City", CITIES)
    Locality = st.selectbox("Locality", LOCALITIES)
    Property_Type = st.selectbox("Property Type", PROPERTY_TYPES)

with c2:
    BHK = st.number_input("BHK", 1, 10, 2)
    Size_in_SqFt = st.number_input("Size (SqFt)", 300, 10000, 1000)
    Price_in_Lakhs = st.number_input("Price (Lakhs)", 1.0, 500.0, 50.0)
    Price_per_SqFt = st.number_input("Price per SqFt", 500.0, 50000.0)

with c3:
    Year_Built = st.number_input("Year Built", 1950, 2025, 2015)
    Age_of_Property = st.number_input("Age of Property", 0, 100, 10)
    Furnished_Status = st.selectbox("Furnished Status", FURNISHED_STATUS)
    Owner_Type = st.selectbox("Owner Type", OWNER_TYPES)

# -----------------------------
# Building & Amenities
# -----------------------------
st.subheader("🏢 Building & Amenities")
c4, c5, c6 = st.columns(3)

with c4:
    Floor_No = st.number_input("Floor Number", 0, 50, 1)
    Total_Floors = st.number_input("Total Floors", 1, 60, 10)
    Parking_Space = st.selectbox("Parking Space", ["Yes", "No"])
    Security = st.selectbox("Security", ["Yes", "No"])

with c5:
    Availability_Status = st.selectbox("Availability Status", ['Under_Construction', 'Ready_to_Move'])
    Facing = st.selectbox("Facing", ORDINAL_COLS["Facing"])
    Public_Transport_Accessibility = st.selectbox(
        "Public Transport Accessibility",
        ORDINAL_COLS["Public_Transport_Accessibility"]
    )

with c6:
    Nearby_Schools = st.number_input("Nearby Schools", 0, 20, 3)
    Nearby_Hospitals = st.number_input("Nearby Hospitals", 0, 20, 2)
    Amenity_Count = st.number_input("Amenity Count", 0, 20, 5)

# -----------------------------
# Locality Intelligence
# -----------------------------
st.subheader("📊 Locality Intelligence")
c7, c8, c9,c10 = st.columns(4)

with c7:
    Locality_Median_Price = st.number_input("Locality Median Price", 0.0)
    Locality_Property_Count = st.number_input( "Locality Property Count", min_value=0, step=1)
with c8:
    Locality_Median_Price_per_sqft = st.number_input("Locality Median Price / SqFt", 0.0)
    Locality_Avg_Age = st.number_input( "Average Property Age in Locality", min_value=0.0)
with c9:
    Investment_Score = st.slider("Investment Score", 0, 100, 50)
    Locality_Avg_BHK = st.number_input("Average BHK in Locality", min_value=0.0)
with c10:
    Locality_Amenity_Density = st.number_input( "Locality Amenity Density", min_value=0.0)
    Locality_Property_Count = st.number_input('Locality_Property_Count',min_value=0.0)

# -----------------------------
# Input DataFrame
# -----------------------------
input_df = pd.DataFrame([{
    'State': State,
    'City': City,
    'Locality': Locality,
    'Property_Type': Property_Type,
    'BHK': BHK,
    'Size_in_SqFt': Size_in_SqFt,
    'Price_in_Lakhs': Price_in_Lakhs,
    'Price_per_SqFt': Price_per_SqFt,
    'Year_Built': Year_Built,
    'Furnished_Status': Furnished_Status,
    'Floor_No': Floor_No,
    'Total_Floors': Total_Floors,
    'Age_of_Property': Age_of_Property,
    'Nearby_Schools': Nearby_Schools,
    'Nearby_Hospitals': Nearby_Hospitals,
    'Public_Transport_Accessibility': Public_Transport_Accessibility,
    'Parking_Space': Parking_Space,
    'Security': Security,
    'Amenities': Amenity_Count,
    'Facing': Facing,
    'Owner_Type': Owner_Type,
    'Availability_Status': Availability_Status,
    'Locality_Median_Price': Locality_Median_Price,
    'Locality_Median_Price_per_sqft': Locality_Median_Price_per_sqft,
    'Investment_Score': Investment_Score,
    'Amenity_Count': Amenity_Count,
    'Locality_Property_Count':Locality_Property_Count, 
    'Locality_Avg_Age' :Locality_Avg_Age,
    'Locality_Avg_BHK' : Locality_Avg_BHK,
    'Locality_Amenity_Density':Locality_Amenity_Density
}])

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🚀 Predict"):
    if mode == "Investment Classification":
        pred = clf_model.predict(input_df)[0]
        prob = clf_model.predict_proba(input_df).max()
        result=""
        if(pred):
            result='Profitable Investement'
        else:
            result='Riski Investment'
        st.success(f"🏷 Investment Category: **{result}**")
        st.info(f"Confidence: **{prob:.2%}**")

    else:
        price = reg_model.predict(input_df)[0]
        st.success(f"💰 Estimated 5-Year Future Price: **₹ {price:,.2f} Lakhs**")















