from pathlib import Path

BASE = Path("data/raw")

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")

write(BASE / "population/population.csv", """
Year,State,AI_AN_Population,Total_Population
2020,Arizona,296529,7151502
2020,New Mexico,219513,2117522
2020,Oklahoma,523549,3959353
2020,Alaska,113953,733391
2020,Montana,79840,1084225
2020,California,649862,39538223
2020,Texas,315944,29145505
2020,Washington,224377,7705281
2020,North Carolina,130806,10439388
2020,Minnesota,119757,5706494
2020,South Dakota,72749,886667
2020,Florida,101295,21538187
2020,Colorado,103467,5773714
2020,Nevada,62079,3104614
2020,Oregon,100422,4237256
2020,Utah,60322,3271616
2020,Michigan,139735,10077331
2020,New York,231594,20201249
2020,Idaho,33746,1839106
2020,New Jersey,49318,9288994
""")

write(BASE / "mortality/mortality.csv", """
Year,State,Condition,AI_AN_Rate_per_100k,Comparator_Rate_per_100k
2020,Arizona,Diabetes,170,75
2020,New Mexico,Diabetes,165,73
2020,Oklahoma,Diabetes,150,70
2020,Alaska,Diabetes,120,68
2020,Montana,Diabetes,115,67
2020,California,Diabetes,130,72
2020,Texas,Diabetes,125,71
2020,Washington,Diabetes,118,69
2020,North Carolina,Diabetes,110,66
2020,Minnesota,Diabetes,105,64
2020,South Dakota,Diabetes,140,68
2020,Florida,Diabetes,100,65
2020,Colorado,Diabetes,98,63
2020,Nevada,Diabetes,108,66
2020,Oregon,Diabetes,102,64
2020,Utah,Diabetes,95,61
2020,Michigan,Diabetes,112,67
2020,New York,Diabetes,97,62
2020,Idaho,Diabetes,94,60
2020,New Jersey,Diabetes,96,61
""")

write(BASE / "missing_persons/missing_persons.csv", """
Year,State,Total_Missing,AI_AN_Missing,AI_AN_Population_Percent
2020,Arizona,1000,90,4.1
2020,New Mexico,700,70,10.4
2020,Oklahoma,1200,95,13.2
2020,Alaska,500,60,15.5
2020,Montana,400,35,6.8
2020,California,3000,85,1.6
2020,Texas,3500,60,1.1
2020,Washington,1100,55,2.9
2020,North Carolina,900,20,1.3
2020,Minnesota,800,30,2.1
2020,South Dakota,450,50,8.2
2020,Florida,2000,18,0.5
2020,Colorado,750,22,1.8
2020,Nevada,650,18,2.0
2020,Oregon,700,28,2.4
2020,Utah,500,15,1.8
2020,Michigan,900,19,1.4
2020,New York,1500,16,1.1
2020,Idaho,400,10,1.8
2020,New Jersey,800,8,0.5
""")

write(BASE / "historical_policy/historical_policy.csv", """
State,Indicator,Value,Definition,Source_Label
Arizona,Relocation_Exposure,8,Policy proxy,Curated
New Mexico,Relocation_Exposure,7,Policy proxy,Curated
Oklahoma,Relocation_Exposure,9,Policy proxy,Curated
Alaska,Relocation_Exposure,5,Policy proxy,Curated
Montana,Relocation_Exposure,6,Policy proxy,Curated
California,Relocation_Exposure,4,Policy proxy,Curated
Texas,Relocation_Exposure,3,Policy proxy,Curated
Washington,Relocation_Exposure,5,Policy proxy,Curated
North Carolina,Relocation_Exposure,4,Policy proxy,Curated
Minnesota,Relocation_Exposure,5,Policy proxy,Curated
South Dakota,Relocation_Exposure,8,Policy proxy,Curated
Florida,Relocation_Exposure,2,Policy proxy,Curated
Colorado,Relocation_Exposure,4,Policy proxy,Curated
Nevada,Relocation_Exposure,5,Policy proxy,Curated
Oregon,Relocation_Exposure,5,Policy proxy,Curated
Utah,Relocation_Exposure,4,Policy proxy,Curated
Michigan,Relocation_Exposure,3,Policy proxy,Curated
New York,Relocation_Exposure,2,Policy proxy,Curated
Idaho,Relocation_Exposure,4,Policy proxy,Curated
New Jersey,Relocation_Exposure,1,Policy proxy,Curated
""")

write(BASE / "environmental/environmental_hazards.csv", """
State,Indicator,Value,Definition,Source_Label
Arizona,Uranium_Exposure_Proxy,9,Env proxy,Curated
New Mexico,Uranium_Exposure_Proxy,8,Env proxy,Curated
Oklahoma,Uranium_Exposure_Proxy,4,Env proxy,Curated
Alaska,Uranium_Exposure_Proxy,5,Env proxy,Curated
Montana,Uranium_Exposure_Proxy,6,Env proxy,Curated
California,Uranium_Exposure_Proxy,3,Env proxy,Curated
Texas,Uranium_Exposure_Proxy,3,Env proxy,Curated
Washington,Uranium_Exposure_Proxy,4,Env proxy,Curated
North Carolina,Uranium_Exposure_Proxy,2,Env proxy,Curated
Minnesota,Uranium_Exposure_Proxy,3,Env proxy,Curated
South Dakota,Uranium_Exposure_Proxy,7,Env proxy,Curated
Florida,Uranium_Exposure_Proxy,1,Env proxy,Curated
Colorado,Uranium_Exposure_Proxy,5,Env proxy,Curated
Nevada,Uranium_Exposure_Proxy,5,Env proxy,Curated
Oregon,Uranium_Exposure_Proxy,3,Env proxy,Curated
Utah,Uranium_Exposure_Proxy,6,Env proxy,Curated
Michigan,Uranium_Exposure_Proxy,2,Env proxy,Curated
New York,Uranium_Exposure_Proxy,1,Env proxy,Curated
Idaho,Uranium_Exposure_Proxy,4,Env proxy,Curated
New Jersey,Uranium_Exposure_Proxy,1,Env proxy,Curated
""")

write(BASE / "boarding_schools/boarding_school_listing.csv", """
State,School_Name,Open_Year,Close_Year,Burial_Site_Indicator
Arizona,School_AZ,1890,1950,1
New Mexico,School_NM,1885,1955,1
Oklahoma,School_OK,1895,1970,1
Alaska,School_AK,1905,1965,0
Montana,School_MT,1892,1958,1
California,School_CA,1901,1951,0
Texas,School_TX,1908,1962,0
Washington,School_WA,1898,1959,0
North Carolina,School_NC,1903,1957,0
Minnesota,School_MN,1891,1961,1
South Dakota,School_SD,1889,1963,1
Florida,School_FL,1910,1950,0
Colorado,School_CO,1904,1956,0
Nevada,School_NV,1902,1954,0
Oregon,School_OR,1897,1953,0
Utah,School_UT,1906,1958,0
Michigan,School_MI,1900,1948,0
New York,School_NY,1912,1952,0
Idaho,School_ID,1907,1955,0
New Jersey,School_NJ,1911,1949,0
""")

Path("config").mkdir(parents=True, exist_ok=True)
Path("config/indicator_weights.json").write_text("""{
  "Relocation_Exposure": 1.0,
  "Uranium_Exposure_Proxy": 1.0,
  "BoardingSchool_Count": 1.0,
  "Mean_BoardingSchool_Duration_Years": 1.0,
  "Max_BoardingSchool_Duration_Years": 1.0,
  "Schools_With_Burial_Site_Flag": 1.0
}
""", encoding="utf-8")

print("Created raw data and config files.")
