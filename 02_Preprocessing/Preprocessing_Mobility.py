import os
import pandas as pd
import numpy as np
from functools import reduce

os.chdir("/Users/sophiemerl/Desktop/Data Analytics in Business/PythonProject")
path = os.getcwd()

######### APPLE
file = "applemobilitytrends-2021-10-23.csv"
apple = pd.read_csv(file)

apple_london = apple[apple["region"] == "London"]
apple_london = apple_london.drop(['region',
                                  'geo_type',
                                  "alternative_name",
                                  "sub-region",
                                  "country"], axis=1)

apple_london = apple_london.melt(id_vars="transportation_type", 
                  var_name="Date", 
                  value_name="Value")

apple_driving = apple_london[apple_london["transportation_type"] == "driving"].drop("transportation_type", axis=1).rename(columns={"Value": "driving"})
apple_transit = apple_london[apple_london["transportation_type"] == "transit"].drop("transportation_type", axis=1).rename(columns={"Value": "transit"})
apple_walking = apple_london[apple_london["transportation_type"] == "walking"].drop("transportation_type", axis=1).rename(columns={"Value": "walking"})

apple = apple_driving.merge(apple_transit, on = "Date").merge(apple_walking, on = "Date").replace(np.nan, -999)

apple['Date'] = pd.to_datetime(apple['Date'], format='%Y-%m-%d')

######### GOOGLE

file = "google_activity_by_London_Borough-2.csv"
google = pd.read_csv(file)

google_london = google.drop(['area_code'], axis=1)

google_city = google_london[google_london["area_name"] == "City of London"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_city",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_city",
                                                                                        "parks_percent_change_from_baseline": "parks_city",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_city",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_city",
                                                                                        "residential_percent_change_from_baseline": "residential_city"}).drop(['area_name'], axis=1)

google_westminster = google_london[google_london["area_name"] == "Westminster"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_westminster",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_westminster",
                                                                                        "parks_percent_change_from_baseline": "parks_westminster",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_westminster",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_westminster",
                                                                                        "residential_percent_change_from_baseline": "residential_westminster"}).drop(['area_name'], axis=1)

google_barking_dagenham = google_london[google_london["area_name"] == "Barking and Dagenham"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_barking/dagenham",
                                                                                                              "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_barking/dagenham",
                                                                                                              "parks_percent_change_from_baseline": "parks_barking/dagenham",
                                                                                                              "transit_stations_percent_change_from_baseline": "transit_barking/dagenham",
                                                                                                              "workplaces_percent_change_from_baseline": "workplaces_barking/dagenham",
                                                                                                              "residential_percent_change_from_baseline": "residential_barking/dagenham"}).drop(['area_name'], axis=1)

google_barnet = google_london[google_london["area_name"] == "Barnet"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_barnet",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_barnet",
                                                                                        "parks_percent_change_from_baseline": "parks_barnet",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_barnet",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_barnet",
                                                                                        "residential_percent_change_from_baseline": "residential_barnet"}).drop(['area_name'], axis=1)


google_bexley = google_london[google_london["area_name"] == "Bexley"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_bexley",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_bexley",
                                                                                        "parks_percent_change_from_baseline": "parks_bexley",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_bexley",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_bexley",
                                                                                        "residential_percent_change_from_baseline": "residential_bexley"}).drop(['area_name'], axis=1)

google_brent = google_london[google_london["area_name"] == "Brent"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_brent",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_brent",
                                                                                        "parks_percent_change_from_baseline": "parks_brent",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_brent",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_brent",
                                                                                        "residential_percent_change_from_baseline": "residential_brent"}).drop(['area_name'], axis=1)

google_bromley = google_london[google_london["area_name"] == "Bromley"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_bromley",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_bromley",
                                                                                        "parks_percent_change_from_baseline": "parks_bromley",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_bromley",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_bromley",
                                                                                        "residential_percent_change_from_baseline": "residential_bromley"}).drop(['area_name'], axis=1)

google_camden = google_london[google_london["area_name"] == "Camden"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_camden",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_camden",
                                                                                        "parks_percent_change_from_baseline": "parks_camden",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_camden",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_camden",
                                                                                        "residential_percent_change_from_baseline": "residential_camden"}).drop(['area_name'], axis=1)

google_croydon = google_london[google_london["area_name"] == "Croydon"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_croydon",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_croydon",
                                                                                        "parks_percent_change_from_baseline": "parks_croydon",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_croydon",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_croydon",
                                                                                        "residential_percent_change_from_baseline": "residential_croydon"}).drop(['area_name'], axis=1)


google_ealing = google_london[google_london["area_name"] == "Ealing"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_ealing",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_ealing",
                                                                                        "parks_percent_change_from_baseline": "parks_ealing",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_ealing",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_ealing",
                                                                                        "residential_percent_change_from_baseline": "residential_ealing"}).drop(['area_name'], axis=1)

google_enfield = google_london[google_london["area_name"] == "Enfield"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_enfield",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_enfield",
                                                                                        "parks_percent_change_from_baseline": "parks_enfield",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_enfield",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_enfield",
                                                                                        "residential_percent_change_from_baseline": "residential_enfield"}).drop(['area_name'], axis=1)

google_hackney = google_london[google_london["area_name"] == "Hackney"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_hackney",
                                                                                        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_hackney",
                                                                                        "parks_percent_change_from_baseline": "parks_hackney",
                                                                                        "transit_stations_percent_change_from_baseline": "transit_hackney",
                                                                                        "workplaces_percent_change_from_baseline": "workplaces_hackney",
                                                                                        "residential_percent_change_from_baseline": "residential_hackney"}).drop(['area_name'], axis=1)

google_hammersmith_fulham = google_london[google_london["area_name"] == "Hammersmith and Fulham"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_hammersmith_fulham",
                                                                                                                  "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_hammersmith_fulham",
                                                                                                                  "parks_percent_change_from_baseline": "parks_hammersmith_fulham",
                                                                                                                  "transit_stations_percent_change_from_baseline": "transit_hammersmith_fulham",
                                                                                                                  "workplaces_percent_change_from_baseline": "workplaces_hammersmith_fulham",
                                                                                                                  "residential_percent_change_from_baseline": "residential_hammersmith_fulham"}).drop(['area_name'], axis=1)

google_haringey = google_london[google_london["area_name"] == "Haringey"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_haringey",
                                                                                          "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_haringey",
                                                                                          "parks_percent_change_from_baseline": "parks_haringey",
                                                                                          "transit_stations_percent_change_from_baseline": "transit_haringey",
                                                                                            "workplaces_percent_change_from_baseline": "workplaces_haringey",
                                                                                            "residential_percent_change_from_baseline": "residential_haringey"}).drop(['area_name'], axis=1)

google_havering = google_london[google_london["area_name"] == "Havering"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_havering",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_havering",
                                                                                      "parks_percent_change_from_baseline": "parks_havering",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_havering",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_havering",
                                                                                      "residential_percent_change_from_baseline": "residential_havering"}).drop(['area_name'], axis=1)

google_harrow = google_london[google_london["area_name"] == "Harrow"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_harrow",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_harrow",
                                                                                      "parks_percent_change_from_baseline": "parks_harrow",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_harrow",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_harrow",
                                                                                      "residential_percent_change_from_baseline": "residential_harrow"}).drop(['area_name'], axis=1)

google_hillingdon = google_london[google_london["area_name"] == "Hillingdon"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_hillingdon",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_hillingdon",
                                                                                      "parks_percent_change_from_baseline": "parks_hillingdon",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_hillingdon",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_hillingdon",
                                                                                      "residential_percent_change_from_baseline": "residential_hillingdon"}).drop(['area_name'], axis=1)

google_hounslow = google_london[google_london["area_name"] == "Hounslow"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_hounslow",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_hounslow",
                                                                                      "parks_percent_change_from_baseline": "parks_hounslow",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_hounslow",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_hounslow",
                                                                                      "residential_percent_change_from_baseline": "residential_hounslow"}).drop(['area_name'], axis=1)

google_islington = google_london[google_london["area_name"] == "Islington"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_islington",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_islington",
                                                                                      "parks_percent_change_from_baseline": "parks_islington",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_islington",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_islington",
                                                                                      "residential_percent_change_from_baseline": "residential_islington"}).drop(['area_name'], axis=1)

google_lambeth = google_london[google_london["area_name"] == "Lambeth"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_lambeth",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_lambeth",
                                                                                      "parks_percent_change_from_baseline": "parks_lambeth",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_lambeth",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_lambeth",
                                                                                      "residential_percent_change_from_baseline": "residential_lambeth"}).drop(['area_name'], axis=1)

google_lewisham = google_london[google_london["area_name"] == "Lewisham"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_lewisham",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_lewisham",
                                                                                      "parks_percent_change_from_baseline": "parks_lewisham",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_lewisham",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_lewisham",
                                                                                      "residential_percent_change_from_baseline": "residential_lewisham"}).drop(['area_name'], axis=1)

google_merton = google_london[google_london["area_name"] == "Merton"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_merton",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_merton",
                                                                                      "parks_percent_change_from_baseline": "parks_merton",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_merton",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_merton",
                                                                                      "residential_percent_change_from_baseline": "residential_merton"}).drop(['area_name'], axis=1)

google_newham = google_london[google_london["area_name"] == "Newham"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_newham",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_newham",
                                                                                      "parks_percent_change_from_baseline": "parks_newham",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_newham",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_newham",
                                                                                      "residential_percent_change_from_baseline": "residential_newham"}).drop(['area_name'], axis=1)

google_redbridge = google_london[google_london["area_name"] == "Redbridge"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_redbridge",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_redbridge",
                                                                                      "parks_percent_change_from_baseline": "parks_redbridge",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_redbridge",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_redbridge",
                                                                                      "residential_percent_change_from_baseline": "residential_redbridge"}).drop(['area_name'], axis=1)

google_richmond = google_london[google_london["area_name"] == "Richmond upon Thames"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_richmond",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_richmond",
                                                                                      "parks_percent_change_from_baseline": "parks_richmond",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_richmond",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_richmond",
                                                                                      "residential_percent_change_from_baseline": "residential_richmond"}).drop(['area_name'], axis=1)

google_southwark = google_london[google_london["area_name"] == "Southwark"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_southwark",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_southwark",
                                                                                      "parks_percent_change_from_baseline": "parks_southwark",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_southwark",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_southwark",
                                                                                      "residential_percent_change_from_baseline": "residential_southwark"}).drop(['area_name'], axis=1)

google_sutton = google_london[google_london["area_name"] == "Sutton"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_sutton",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_sutton",
                                                                                      "parks_percent_change_from_baseline": "parks_sutton",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_sutton",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_sutton",
                                                                                      "residential_percent_change_from_baseline": "residential_sutton"}).drop(['area_name'], axis=1)

google_tower_hamlets = google_london[google_london["area_name"] == "Tower Hamlets"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_tower_hamlets",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_tower_hamlets",
                                                                                      "parks_percent_change_from_baseline": "parks_tower_hamlets",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_tower_hamlets",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_tower_hamlets",
                                                                                      "residential_percent_change_from_baseline": "residential_tower_hamlets"}).drop(['area_name'], axis=1)

google_waltham_forest = google_london[google_london["area_name"] == "Waltham Forest"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_waltham_forest",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_waltham_forest",
                                                                                      "parks_percent_change_from_baseline": "parks_waltham_forest",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_waltham_forest",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_waltham_forest",
                                                                                      "residential_percent_change_from_baseline": "residential_waltham_forest"}).drop(['area_name'], axis=1)

google_wandsworth = google_london[google_london["area_name"] == "Wandsworth"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_wandsworth",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_wandsworth",
                                                                                      "parks_percent_change_from_baseline": "parks_wandsworth",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_wandsworth",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_wandsworth",
                                                                                      "residential_percent_change_from_baseline": "residential_wandsworth"}).drop(['area_name'], axis=1)

google_greenwich = google_london[google_london["area_name"] == "Greenwich"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_greenwich",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_greenwich",
                                                                                      "parks_percent_change_from_baseline": "parks_greenwich",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_greenwich",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_greenwich",
                                                                                      "residential_percent_change_from_baseline": "residential_greenwich"}).drop(['area_name'], axis=1)

google_kensington_chelsea = google_london[google_london["area_name"] == "Kensington and Chelsea"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_kensington_chelsea",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_kensington_chelsea",
                                                                                      "parks_percent_change_from_baseline": "parks_kensington_chelsea",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_kensington_chelsea",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_kensington_chelsea",
                                                                                      "residential_percent_change_from_baseline": "residential_kensington_chelsea"}).drop(['area_name'], axis=1)

google_kingston = google_london[google_london["area_name"] == "Kingston upon Thames"].rename(columns={"retail_and_recreation_percent_change_from_baseline": "retail_recreation_kingston",
                                                                                      "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_kingston",
                                                                                      "parks_percent_change_from_baseline": "parks_kingston",
                                                                                      "transit_stations_percent_change_from_baseline": "transit_kingston",
                                                                                      "workplaces_percent_change_from_baseline": "workplaces_kingston",
                                                                                      "residential_percent_change_from_baseline": "residential_kingston"}).drop(['area_name'], axis=1)

boroughs = [google_city,
            google_westminster,
            google_barking_dagenham,
            google_barnet,
            google_bexley,
            google_brent,
            google_bromley,
            google_camden,
            google_croydon,
            google_ealing,
            google_enfield,
            google_hackney,
            google_hammersmith_fulham,
            google_haringey,
            google_harrow,
            google_havering,
            google_hillingdon,
            google_hounslow,
            google_islington,
            google_lambeth,
            google_lewisham,
            google_merton,
            google_newham,
            google_redbridge,
            google_richmond,
            google_southwark,
            google_sutton,
            google_tower_hamlets,
            google_waltham_forest,
            google_wandsworth,
            google_greenwich,
            google_kensington_chelsea,
            google_kingston]


google_boroughs = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='outer'), boroughs).replace(np.nan, -999)


google_boroughs = google_boroughs.rename(columns={"date": "Date"})

google_boroughs['Date'] = pd.to_datetime(google_boroughs['Date'], format='%Y-%m-%d')

######### RESTAURANTS

file = "2020-2021vs2019_Seated_Diner_Data-2.csv"
restaurants = pd.read_csv(file)

restaurants_london = restaurants[restaurants["Name"] == "London"]

restaurants_london = restaurants_london.melt(id_vars="Name", 
                  var_name="Date", 
                  value_name="Value").drop("Name", axis=1).rename(columns={"Value": "dinner_seated"})

restaurants_london = restaurants_london.iloc[1:,:]

restaurants_london['Date'] = pd.to_datetime(restaurants_london['Date'], format='%Y/%m/%d')

######### WEATHER

file = "weather.csv"
weather = pd.read_csv(file)

weather = weather.rename(columns={"date": "Date"}).drop("tsun", axis=1).drop("snow", axis=1)

weather['Date'] = pd.to_datetime(weather['Date'], format='%Y-%m-%d')

######### CITYMAPPER/ PUBLIC TRANSPORT 
#Note: the actual value (value) not the 7-day average (roll) was used --> not adjusted to intraweek fluctuations

file = "citymapper_v1.csv"
citymapper = pd.read_csv(file)
citymapper = citymapper.rename(columns={"date": "Date"})
citymapper['Date'] = pd.to_datetime(citymapper['Date'], format='%Y/%m/%d')

Tube = citymapper[citymapper["source"] == "TfL Tube"].drop("roll", axis=1).drop("activity", axis=1).drop("source", axis=1).rename(columns={"value": "Tube"})
Bus = citymapper[citymapper["source"] == "TfL Bus"].drop("roll", axis=1).drop("activity", axis=1).drop("source", axis=1).rename(columns={"value": "Bus"})
Citymapper = citymapper[citymapper["source"] == "Citymapper"].drop("roll", axis=1).drop("activity", axis=1).drop("source", axis=1).rename(columns={"value": "Citymapper"})

tfl_citymapper = [Tube, Bus, Citymapper] #Google_transit already in google_boroughs

citymapper_public = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), tfl_citymapper).replace(np.nan, -999)

######### MERGE

data_sets = [google_boroughs,
             apple,
             citymapper_public,
             weather,
             restaurants_london]

data_sets_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), data_sets)

data_sets_merged.to_csv(r'London_301021.csv')











