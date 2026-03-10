Merged Datasets: Bird Radar + KNMI Weather (Station 286)
=========================================================
refer:  join_knmi_286.py
Files:
  train_with_knmi_286.csv  –  train.csv enriched with hourly weather
  test_with_knmi_286.csv   –  test.csv  enriched with hourly weather

Source weather data: KNMI station 286 (Nieuw Beerta), hourly observations.

Join method: Each radar track's temporal midpoint is matched to the next
KNMI hour-end bucket using a forward asof join (tolerance: 1 hour).
KNMI "HH" denotes the end of the measurement interval (HH=15 → 14:00–15:00).

Columns dropped: P, VV, N, WW, M, R, S, O, Y (station 286 does not report these).
Remaining weather columns were renamed to human-readable form.
Wind direction was additionally encoded as sin/cos plus a variable-wind flag.

Added columns:
  knmi_286_hour_end_timestamp_utc        – matched KNMI hour-end timestamp (UTC)
  knmi_286_station_id                    – KNMI station number (always 286)
  knmi_286_wind_direction_degrees        – hourly mean wind direction in degrees (990 = variable)
  knmi_286_hourly_mean_wind_speed_mps    – hourly mean wind speed (m/s)
  knmi_286_wind_speed_at_observation_mps – wind speed at the moment of observation (m/s)
  knmi_286_max_wind_gust_mps             – maximum wind gust in the past hour (m/s)
  knmi_286_air_temperature_c             – air temperature at 1.5 m (°C)
  knmi_286_min_air_temperature_last_6h_c – minimum temperature in the preceding 6 hours (°C)
  knmi_286_dew_point_temperature_c       – dew point temperature (°C)
  knmi_286_sunshine_duration_hours        – sunshine duration in the past hour (hours, 0–1)
  knmi_286_global_radiation_j_cm2        – global radiation (J/cm²)
  knmi_286_precipitation_duration_hours  – precipitation duration in the past hour (hours, 0–1)
  knmi_286_precipitation_amount_mm       – hourly precipitation sum (mm); -0.1 = trace (< 0.05 mm)
  knmi_286_relative_humidity_percent     – relative humidity (%)
  knmi_286_weather_indicator_code        – KNMI present-weather indicator (0–9 code)
  knmi_286_wind_dir_sin                  – sin(wind_direction) for cyclical encoding
  knmi_286_wind_dir_cos                  – cos(wind_direction) for cyclical encoding
  knmi_286_wind_dir_variable             – 1 if wind direction was variable (DD=990), else 0
  knmi_286_match_time_difference_minutes – minutes between track midpoint and matched KNMI hour
