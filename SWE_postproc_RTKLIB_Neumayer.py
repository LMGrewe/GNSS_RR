""" Run RTKLib automatically for differential GNSS post processing and SWE estimation at the surroundings of the NeumayerIII station
http://www.rtklib.com/

Reference:  - Steiner et al., Combined GNSS reflectometry/refractometry for continuous in situ surface mass balance estimation on an Antarctic ice shelf, AGU, 2022.
            - T.Takasu, RTKLIB: Open Source Program Package for RTK-GPS, FOSS4G 2009 Tokyo, Japan, November 2, 2009
            - Thomas Nischan (2016): GFZRNX - RINEX GNSS Data Conversion and Manipulation Toolbox. GFZ Data Services. https://doi.org/10.5880/GFZ.1.1.2016.002

Python 3.10
input:  - GNSS config file (.conf)
        - GNSS rover file (rinex)
        - GNSS base file (rinex)
        - GNSS navigation ephemerides file (.nav); ftp://cddis.nasa.gov/archive/gnss/data/daily/YYYY/brdc/brdc*.yy[nge].gz
        - GNSS precise ephemerides file (.eph/.sp3); ftp://ftp.aiub.unibe.ch/YYYY_M/COD*.EPH_M.zip

output: - position (.pos) file; (UTC, E, N, U)
        - plots (SWE timeseries, DeltaSWE timeseries, scatter plots)

requirements:   - rtklib (v2.4.3 b34, https://www.rtklib.com/)
                - gfzrnx (https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577894)
                - path to all programs added to the system environment variables

created by: L. Steiner (ORCID: 0000-0002-4958-0849) - original codebase: https://github.com/lasteine/GNSS_RR.git
revised and expanded by: L.M. Grewe (ORCID: 0009-0009-6533-3432)
created on: 17.05.2022
last updated on: 27.10.2023
"""

# IMPORT modules
import os
import pandas as pd
import functions as f
import numpy as np
import datetime as dt

# CHOOSE: DEFINE data paths, file names (base, rover, navigation orbits, precise orbits, config), time interval, and processing steps
src_path = '//smb.isibhv.dmawi.de/projects/p_gnss/DATA/'                                                                # data source path at AWI server (data copied from Antarctica via O2A)
src_path_mob = '//smb.isibhv.dmawi.de/projects/mob/Neumayer/data/Rohdaten/gnss/'                                        # source path to rover data of meteorologic observatory
dest_path = 'C:/Users/.../processing_directory/'                                                                        # local destination path for processing
backup_path = '...'                                                                                                     # directory path to where the whole processing directory (dest_path-directory) is copied
laser_path = '//smb.isibhv.dmawi.de/projects/mob/Neumayer/data/Rohdaten/shm/'                                           # data source path at AWI server for snow accumulation laser sensor from AWI MetObs
mob_path = '//smb.isibhv.dmawi.de/projects/mob/Neumayer/data/val/'                                                      # data source path at AWI server to meteorologic observations from AWI MetObs (mob)
synop_path = '//smb.isibhv.dmawi.de/projects/mob/Neumayer/data/obs/archive/'                                            # data source path at AWI server to meteorologic observations SYNOPS from AWI MetObs (mob)
buoy_url = 'https://data.meereisportal.de/data/buoys/processed/2017S54_data.zip'                                        # data path for snow buoy data close to Spuso from sea ice physics group
reflecto_sol_src_path = '.../reflectometry_solutions/'                                                                  # data source path at the Ubuntu localhost (GNSS reflectometry processing location)
rover = 'ReachM2_sladina-raw_'                                                                                          # 'NMER' or '3393' (old Emlid: 'ReachM2_sladina-raw_')
rover_name = 'NMER_original'                                                                                            # 'NMER' or 'NMER_original' or 'NMLR'
receiver = 'NMER'                                                                                                       # 'NMER' or 'NMLB' or 'NMLR'
base_name = 'NMLB'                                                                                                      # prefix of base rinex observation files, e.g. station name
LB = '3387'                                                                                                             # rinex file name prefix for Leica Base receiver
JB = 'nmsh'                                                                                                             # rinex file name prefix for JAVAD Base receiver
nav = '3387'                                                                                                            # navigation file name prefix for broadcast ephemerides files
sp3 = 'COD'                                                                                                             # navigation file name prefix for precise ephemerides files
ti_int = '900'                                                                                                          # processing time interval (seconds)
resolution = '15min'                                                                                                    # processing resolution (minutes)
options_LBLR = 'rtkpost_options_Leica_statisch_multisystemfrequency_neumayer_900_15'                                    # name of RTKLIB configuration file (.conf) for Leica high-end receiver and base
options_LBUR = 'rtkpost_options_Emlid_statisch_multisystemfrequency_neumayer_900_15'                                    # name of RTKLIB configuration file (.conf) for Emlid low-cost receiver and high-end Leica base
options_JBLR = 'rtkpost_options_Leica_JAVAD_statisch_multisystemfrequency_neumayer_900_15'                              # name of RTKLIB configuration file (.conf) for Leica high-end receiver and JAVAD high-end base
options_JBUR = 'rtkpost_options_Emlid_JAVAD_statisch_multisystemfrequency_neumayer_900_15'                              # name of RTKLIB configuration file (.conf) for Emlid low-cost receiver and JAVAD high-end base
rnx_version = '3.03'                                                                                                    # used rinex version for merging and processing (with gfzrnx and RTKLIB)
ending = ''                                                                                                             # file name suffix if needed: e.g., a variant of the processing '_eleambmask15', '_noglonass'
acc_y_lim = (-200, 1600)                                                                                                # y-axis limit for accumulation plots
delta_acc_y_lim = (-400, 1000)                                                                                          # y-axis limit for delta accumulation plots
swe_y_lim = (-170, 810)                                                                                                 # y-axis limit for water equivalent plots
delta_swe_y_lim = (-200, 600)                                                                                           # y-axis limit for delta water equivalent plots
xlim_dates = dt.date(2021, 11, 1), dt.date(2023, 8, 31)                                                                 # time series date limits to plot on x-axis
baseline_length_Up = {'LBLR': -3250, 'LBUR': -3250, 'JBLR': -2360, 'JBUR': -2360}                                       # manually measured height (Up-component) difference between base and rover (before snow mast heightening)
cal_date = '2022-07-24'                                                                                                 # calibration date for snow density estimation
yy_LBLR = '21'                                                                                                          # initial year of observations - used for beginning of reference data
new_snow = [[2, 0.01], [2, 0.03], [6, 0.03], [6, 0.05], [14, 0.05], [30, 0.05], [30, 0.1], [60, 0.1], [90, 0.1], [120, 0.1]] # For calculating the snow height, SWE and density of a freshly accumulated snow layer within different pairs of specified [intervals (in days), minimum snow height (in m)]
# create plots (True) or not (False)
plot_0_sol_quality = False                                                                                              # plot number of satellites, ambiguity resolution and daily noise over fixed ambiguities
plot_1_filtering = False                                                                                                # plot each filtering step for all basleines
plot_2_SWE_GNSS = False                                                                                                 # plot SWE of GNSS refractometry and reference SWE (high-end and low-cost)
plot_2_SWE_deviation = False                                                                                            # plot linear fit and deviation between different baseline solutions and between GNSS refractometry and reference SWE
plot_3_GNSS_MOB = False                                                                                                 # plot GNSS observations with meteorologic observations (temperature, sun indicator, present weather: snowfall and snowdrift, windspeed, wind direction)
plot_4_dsh = False                                                                                                      # plot snow height of GNSS-IR and reference sensors
plot_4_dsh_deviation = False                                                                                            # plot deviation between GNSS-IR and reference observations
plot_4_dsh_fit = False                                                                                                  # plot and linear regression between GNSS-IR and reference observations
plot_5_GNSS_RR = False                                                                                                  # plot GNSS refractometry SWE and GNSS-IR snow height together
plot_5_refracto_over_reflecto = False                                                                                   # plot GNSS refractometry SWE versus GNSS-IR snow height and exponential fit
plot_6_density = False                                                                                                  # plot GNSS-RR density time series and reference measurements
plot_6_density_error = False                                                                                            # plot GNSS-RR standard deviation versus snow height (derived by error propagation)
plot_6_density_deviation = False                                                                                        # plot deviation between GNSS-RR and reference densities
plot_7_density_over_dsh = False                                                                                         # plot GNSS-RR density versus snow height and different fits
plot_8_footprint = False                                                                                                # plot GNSS refractometry footprint over snow height for different angles of incidence
plot_9_new_snow_layer = False                                                                                           # plot height, mass and density of a freshly fallen snow layer
save_plots = False                                                                                                      # show (False) or save (True) plots
total_backup = False                                                                                                    # True: copy all new data to server for backup, False: do not copy
solplot_backup = False                                                                                                  # True: copy all new solution files and plots to server for backup, False: do not copy



''' 0. Preprocess data '''

# create processing directory
os.makedirs(dest_path, exist_ok=True)

# copy & uncompress new rinex files (NMLB + all orbits, NMLR, NMER) to processing folder 'data_neumayer/'
end_mjd_ER = f.copy_rinex_files(src_path + 'id8282_refractolow/', dest_path + 'temp_NMER/', receiver='NMER', copy=True,
                                parent=True, hatanaka=True, move=True, delete_temp=True)  # for emlid rover: NMER
end_mjd_LR = f.copy_rinex_files(src_path + 'id8281_refracto/', dest_path + 'temp_NMLR/', receiver='NMLR', copy=True,
                                parent=True, hatanaka=True, move=True, delete_temp=True)  # for leica rover: NMLR
end_mjd_LB = f.copy_rinex_files(src_path + 'id8283_reflecto/', dest_path + 'temp_NMLB/', receiver='NMLB', copy=True,
                                parent=True, hatanaka=True, move=True, delete_temp=True)  # for leica base: NMLB
end_mjd_JB = f.copy_rinex_files(src_path_mob, dest_path + 'temp_NMSH/', receiver='nmsh', copy=True,
                                parent=True, hatanaka=True, move=True, delete_temp=True)  # for javad base: nmsh

# merge all split observation and navigation files of Leica Receivers
df_LR_merged, df_LB_merged = f.merge_split_Leica(dest_path, delete_temp_merge=True)

# check available solution data (to only further process new data and reprocess only already processed merged Files)
# -> first mjd in 2021 here: 59544.0
yy_LBUR, start_mjd_LBUR = f.get_sol_yeardoy(dest_path, resolution, receiver='NMER', base='LB')
yy_LBLR, start_mjd_LBLR = f.get_sol_yeardoy(dest_path, resolution, receiver='NMLR', base='LB')
yy_JBUR, start_mjd_JBUR = f.get_sol_yeardoy(dest_path, resolution, receiver='NMER', base='JB')
yy_JBLR, start_mjd_JBLR = f.get_sol_yeardoy(dest_path, resolution, receiver='NMLR', base='JB')

# for proceeding manually without preprocessing: calculate start/end mjd using a given start/end date
# start_mjd_LBLR, end_mjd_LR= f.get_mjd_int(2021, 11, 12, 2023, 2, 28)
# start_mjd_LBUR, end_mjd_ER = start_mjd_LBLR, end_mjd_LR



''' 1. Run RTKLib automatically (instead of RTKPost Gui manually)'''

# "re-process" all days again that are newly merged and of which solutions already exist (needed IF code was run in an earlier version without f.merge_splitted_Leica)
f.process_merged_Leica(df_LR_merged, df_LB_merged, dest_path, 59560.0, start_mjd_LBUR, start_mjd_LBLR, ti_int, LB, nav, sp3, resolution, ending, options_LBUR, options_LBLR, 'LB')

# process data using RTKLIB post processing command line tool 'rnx2rtkp' for a specific year and a range of day of years (doys)
f.automate_rtklib_pp(dest_path, 'NMER', start_mjd_LBUR, end_mjd_ER, ti_int, LB, nav, sp3, resolution, ending, options_LBUR, rover_name='NMER', base_name='LB')
f.automate_rtklib_pp(dest_path, '3393', start_mjd_LBLR, end_mjd_LR, ti_int, LB, nav, sp3, resolution, ending, options_LBLR, rover_name='NMLR', base_name='LB')
f.automate_rtklib_pp(dest_path, 'NMER', start_mjd_JBUR, end_mjd_ER, ti_int, JB, nav, sp3, resolution, ending, options_JBUR, rover_name='NMER', base_name='JB')
f.automate_rtklib_pp(dest_path, '3393', start_mjd_JBLR, end_mjd_LR, ti_int, JB, nav, sp3, resolution, ending, options_JBLR, rover_name='NMLR', base_name='JB')




''' 2. Get RTKLib ENU solution files '''

# read all RTKLib ENU solution files (daily) and store them in one dataframe for whole season
df_enu_LBUR = f.get_rtklib_solutions(dest_path, 'NMER', resolution, ending, header_length=26, base_name='LB')
df_enu_LBLR = f.get_rtklib_solutions(dest_path, 'NMLR', resolution, ending, header_length=26, base_name='LB')
df_enu_JBUR = f.get_rtklib_solutions(dest_path, 'NMER', resolution, ending, header_length=26, base_name='JB')
df_enu_JBLR = f.get_rtklib_solutions(dest_path, 'NMLR', resolution, ending, header_length=26, base_name='JB')




''' 3. Filter and clean ENU solution data '''

# filter and clean ENU solution data (outlier filtering, median filtering, adjustments for observation mast heightening) and store results in pickle and .csv
fil_df_LBUR, u_LBUR, u_clean_LBUR, swe_unfil_LBUR, swe_gnss_LBUR, std_gnss_LBUR, swe_gnss_daily_LBUR, std_gnss_daily_LBUR, std_gnss_p_LBUR, date_of_min_LBUR = f.filter_rtklib_solutions(
    dest_path, 'NMER', 'LB', baseline_length_Up['LBUR'], resolution, df_enu=df_enu_LBUR, ambiguity=1, threshold=1.9, window='D', ending=ending)
fil_df_LBLR, u_LBLR, u_clean_LBLR, swe_unfil_LBLR, swe_gnss_LBLR, std_gnss_LBLR, swe_gnss_daily_LBLR, std_gnss_daily_LBLR, std_gnss_p_LBLR, date_of_min_LBLR = f.filter_rtklib_solutions(
    dest_path, 'NMLR', 'LB', baseline_length_Up['LBLR'], resolution, df_enu=df_enu_LBLR, ambiguity=1, threshold=3, window='D', ending=ending)
fil_df_JBUR, u_JBUR, u_clean_JBUR, swe_unfil_JBUR, swe_gnss_JBUR, std_gnss_JBUR, swe_gnss_daily_JBUR, std_gnss_daily_JBUR, std_gnss_p_JBUR, date_of_min_JBUR = f.filter_rtklib_solutions(
    dest_path, 'NMER', 'JB', baseline_length_Up['JBUR'], resolution, df_enu=df_enu_JBUR, ambiguity=1, threshold=1.9, window='D', ending=ending)
fil_df_JBLR, u_JBLR, u_clean_JBLR, swe_unfil_JBLR, swe_gnss_JBLR, std_gnss_JBLR, swe_gnss_daily_JBLR, std_gnss_daily_JBLR, std_gnss_p_JBLR, date_of_min_JBLR = f.filter_rtklib_solutions(
    dest_path, 'NMLR', 'JB', baseline_length_Up['JBLR'], resolution, df_enu=df_enu_JBLR, ambiguity=1, threshold=3, window='D', ending=ending)

# calculate GNSS refractometry footprint for different angle of incidence over snow height
GNSS_refr_radius_5 = f.calc_footprint(5)
GNSS_refr_radius_90 = f.calc_footprint(90)




''' 4. Read and filter reference sensors data '''

# import / read laser reference data starting from 01.11.2021
laser = f.read_laser_observations(dest_path, laser_path, yy_LBLR, laser_pickle='nm_laser')
# read all other reference sensors data
manual, manual_NEW, ipol, buoy, poles, mob, synop = f.read_reference_data(dest_path, laser_path, mob_path, synop_path, yy_LBLR, url=buoy_url, read_manual=True, read_buoy=True, read_poles=True, read_laser=False, read_mob=True, read_synop=True, mob_pickle='nm_mob', synop_pickle='nm_synop')
# filter laser data and calculate swe with once constant density of 402 kg/m² and once with the interpolated mnaual density measurements (ipol)
laser_filtered = f.filter_laser_observations(ipol, laser, threshold=1)
# resample laser sensor data to daily and 15min intervals
laser_daily = f.resample_ref_obs(laser_filtered, interval='D')
laser_15min = f.resample_ref_obs(laser_filtered, interval='15min')

# set the GNSS-RR installation date, 16.11.2021 to zero accumulation above antenna level
laser_filtered.dsh = laser_filtered.dsh - laser_daily.dsh.interpolate()['2021-11-16']
laser_daily.dsh = laser_daily.dsh - laser_daily.dsh.interpolate()['2021-11-16']
# get snow height above antenna on date of GNSS-IR observation start (26.11.2021)
Acc_211126 = laser_daily.dsh.interpolate()['2021-11-26']
# Calibration: set the accumulation level of other reference sensors to the actual snow height above antenna
poles[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']] = poles[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']] + Acc_211126
buoy[['dsh1', 'dsh2', 'dsh3', 'dsh4']] = buoy[['dsh1', 'dsh2', 'dsh3', 'dsh4']] + laser_daily.dsh.interpolate()['2021-11-27']

# Calculate mean and standard deviation of snow height measurements (shm) and dSWE of stake field ("pole") and buoy observations
poles, buoy = f.get_mean_and_std_deviation(poles, buoy)
# resample all reference sensors data to daily intervals
buoy_daily = f.resample_ref_obs(buoy, interval='D')
poles_daily = f.resample_ref_obs(poles, interval='D')
mob_daily = f.resample_ref_obs(mob, interval='D')




''' 5. Calculate differences, linear regressions & RMSE between GNSS refractometry and reference data '''

# Solution Quality Control
amb_std_LBLR, amb_std_fit_LBLR, func_exp_LBLR = f.solution_control(fil_df_LBLR.amb_state, std_gnss_daily_LBLR, 90)
amb_std_LBUR, amb_std_fit_LBUR, func_exp_LBUR = f.solution_control(fil_df_LBUR.amb_state, std_gnss_daily_LBUR, 90)
amb_std_JBLR, amb_std_fit_JBLR, func_exp_JBLR = f.solution_control(fil_df_JBLR.amb_state, std_gnss_daily_JBLR, 90)
amb_std_JBUR, amb_std_fit_JBUR, func_exp_JBUR = f.solution_control(fil_df_JBUR.amb_state, std_gnss_daily_JBUR, 90)

# calculate statistical values, representing the difference between the different baseline solutions
# cross correlation
swe_gnss_daily_LBLR.corr(swe_gnss_daily_JBLR)     # between high-end rover solutions
swe_gnss_daily_LBLR.corr(swe_gnss_daily_LBUR)     # between Leica base solutions
swe_gnss_daily_JBLR.corr(swe_gnss_daily_JBUR)     # between JABAD base solutions
swe_gnss_daily_LBUR.corr(swe_gnss_daily_JBUR)     # between low-cost rover solutions
# R², deviation range, variance, and standard deviation (RMSE)
deviation_JB2LB_he, deviation_p_JB2LB_he = f.calculate_stats('LBLR - JBLR, high-end solutions', swe_gnss_daily_LBLR, swe_gnss_daily_JBLR)
deviation_UR2LR_LB, deviation_p_UR2LR_LB = f.calculate_stats('LBLR - LBUR, Leica base solutions', swe_gnss_daily_LBLR, swe_gnss_daily_LBUR)
deviation_UR2LR_JB, deviation_p_UR2LR_JB = f.calculate_stats('JBLR - JBUR, JAVAD base solutions', swe_gnss_daily_JBLR, swe_gnss_daily_JBUR)
deviation_JB2LB_lc, deviation_p_JB2LB_lc = f.calculate_stats('LBUR - JBUR, low-cost solutions', swe_gnss_daily_LBUR, swe_gnss_daily_JBUR)
# ...for high-end solutions before and after 15.02.2022
deviation_JB2LB_he_bf, deviation_p_JB2LB_he_bf = f.calculate_stats('LBLR - JBLR, high-end solutions', swe_gnss_daily_LBLR[swe_gnss_daily_LBLR.index < '2022-02-15'], swe_gnss_daily_JBLR[swe_gnss_daily_JBLR.index < '2022-02-15'])
deviation_JB2LB_he_af, deviation_p_JB2LB_he_af = f.calculate_stats('LBLR - JBLR, high-end solutions', swe_gnss_daily_LBLR[swe_gnss_daily_LBLR.index >= '2022-02-15'], swe_gnss_daily_JBLR[swe_gnss_daily_JBLR.index < '2022-02-15'])

# calculate and plot linear fit and statistics between each baseline solution
LBLR_JBLR_R, LBLR_JBLR_R_p = f.dependency(swe_gnss_daily_LBLR, 'mass (LBLR) [kg/m²]', swe_gnss_daily_JBLR, 'mass (JBLR) [kg/m²]', plot_2_SWE_deviation, '2e1c_dependence_JB2LB_highend', save_plots, dest_path, color='purple', fig_title='a) Leica Rover', ax_lim=(0, 700), tick_interval=100)
LBLR_LBUR_R, LBLR_LBUR_R_p = f.dependency(swe_gnss_daily_LBLR, 'mass (LBLR) [kg/m²]', swe_gnss_daily_LBUR, 'mass (LBUR) [kg/m²]', plot_2_SWE_deviation, '2e2c_dependence_UR2LR_LB', save_plots, dest_path, color='indianred', fig_title='e) Leica Base', ax_lim=(0, 700), tick_interval=100)
JBLR_JBUR_R, JBLR_JBUR_R_p = f.dependency(swe_gnss_daily_JBLR, 'mass (JBLR) [kg/m²]', swe_gnss_daily_JBUR, 'mass (JBUR) [kg/m²]', plot_2_SWE_deviation, '2e3c_dependence_UR2LR_JB', save_plots, dest_path, color='cornflowerblue', fig_title='f) JAVAD Base', ax_lim=(0, 700), tick_interval=100)
#swe_gnss_daily_LBUR.loc[swe_gnss_daily_LBUR[0:] < 0] = np.nan
LBUR_JBUR_R, LBUR_JBUR_R_p = f.dependency(swe_gnss_daily_LBUR, 'mass (LBUR) [kg/m²]', swe_gnss_daily_JBUR, 'mass (JBUR) [kg/m²]', plot_2_SWE_deviation, '2e4c_dependence_JB2LB_lowcost', save_plots, dest_path, color='chocolate', fig_title='c) U-blox Rover', ax_lim=(0, 700), tick_interval=100)
# from 15.02.2022
LBLR_JBLR_R, LBLR_JBLR_R_p = f.dependency(swe_gnss_daily_LBLR[swe_gnss_daily_LBLR.index >= '2022-02-15'], 'mass (LBLR) [kg/m²]', swe_gnss_daily_JBLR[swe_gnss_daily_JBLR.index >= '2022-02-15'], 'mass (JBLR) [kg/m²] ', plot_2_SWE_deviation, '2e1d_dependence_JB2LB_highend_1502', save_plots, dest_path, color='purple', fig_title='b) Leica Rover', ax_lim=(0, 700), tick_interval=100)
LBUR_JBUR_R, LBUR_JBUR_R_p = f.dependency(swe_gnss_daily_LBUR[swe_gnss_daily_LBUR.index >= '2022-02-15'], 'mass (LBUR) [kg/m²]', swe_gnss_daily_JBUR[swe_gnss_daily_JBUR.index >= '2022-02-15'], 'mass (JBUR) [kg/m²] ', plot_2_SWE_deviation, '2e4d_dependence_JB2LB_lowcost_1502', save_plots, dest_path, color='chocolate', fig_title='d) U-blox Rover', ax_lim=(0, 700), tick_interval=100)

# calculate and plot linear fit and calculate statistical values, representing the difference between the baseline solutions and reference sensors
# GNSS refractometry vs. (laser * snow pit)
SWE_LBLR_lasermanual_R, SWE_LBLR_lasermanual_R_p, = f.dependency(laser_daily.dswe, 'mass (laser * snow pit) [kg/m²]', swe_gnss_daily_LBLR.interpolate(), 'mass (LBLR) [kg/m²]', plot_2_SWE_deviation, '2f1a_dependence_SWE_ref2LBLR', save_plots, dest_path, color='crimson', fig_title='a)', ax_lim=(0, 800), tick_interval=100)
SWE_LBUR_lasermanual_R, SWE_LBUR_lasermanual_R_p, = f.dependency(laser_daily.dswe, 'mass (laser * snow pit) [kg/m²]', swe_gnss_daily_LBUR.interpolate(), 'mass (LBUR) [kg/m²]', plot_2_SWE_deviation, '2f1b_dependence_SWE_ref2LBUR', save_plots, dest_path, color='salmon', fig_title='c)', ax_lim=(0, 800), tick_interval=100)
SWE_JBLR_lasermanual_R, SWE_JBLR_lasermanual_R_p, = f.dependency(laser_daily.dswe, 'mass (laser * snow pit) [kg/m²]', swe_gnss_daily_JBLR.interpolate(), 'mass (JBLR) [kg/m²]', plot_2_SWE_deviation, '2f1c_dependence_SWE_ref2JBLR', save_plots, dest_path, color='dodgerblue', fig_title='b)', ax_lim=(0, 800), tick_interval=100)
SWE_JBUR_lasermanual_R, SWE_JBUR_lasermanual_R_p, = f.dependency(laser_daily.dswe, 'mass (laser * snow pit) [kg/m²]', swe_gnss_daily_JBUR.interpolate(), 'mass (JBUR) [kg/m²]', plot_2_SWE_deviation, '2f1d_dependence_SWE_ref2JBUR', save_plots, dest_path, color='deepskyblue', fig_title='d)', ax_lim=(0, 800), tick_interval=100)
# GNSS refractometry vs. (laser * const value)
SWE_LBLR_laserconst_R, SWE_LBLR_laserconst_R_p, = f.dependency(laser_daily.dswe_const, 'mass (laser * 408 kg/m³) [kg/m²]', swe_gnss_daily_LBLR.interpolate(), 'mass (LBLR) [kg/m²]', plot_2_SWE_deviation, '2f2a_dependence_SWE_ref2LBLR', save_plots, dest_path, color='crimson', fig_title='a)', ax_lim=(0, 800), tick_interval=100)
SWE_LBUR_laserconst_R, SWE_LBUR_laserconst_R_p, = f.dependency(laser_daily.dswe_const, 'mass (laser * 408 kg/m³) [kg/m²]', swe_gnss_daily_LBUR.interpolate(), 'mass (LBUR) [kg/m²]', plot_2_SWE_deviation, '2f2b_dependence_SWE_ref2LBUR', save_plots, dest_path, color='salmon', fig_title='c)', ax_lim=(0, 800), tick_interval=100)
SWE_JBLR_laserconst_R, SWE_JBLR_laserconst_R_p, = f.dependency(laser_daily.dswe_const, 'mass (laser * 408 kg/m³) [kg/m²]', swe_gnss_daily_JBLR.interpolate(), 'mass (JBLR) [kg/m²]', plot_2_SWE_deviation, '2f2c_dependence_SWE_ref2JBLR', save_plots, dest_path, color='dodgerblue', fig_title='b)', ax_lim=(0, 800), tick_interval=100)
SWE_JBUR_laserconst_R, SWE_JBUR_laserconst_R_p, = f.dependency(laser_daily.dswe_const, 'mass (laser * 408 kg/m³) [kg/m²]', swe_gnss_daily_JBUR.interpolate(), 'mass (JBUR) [kg/m²]', plot_2_SWE_deviation, '2f2d_dependence_SWE_ref2JBUR', save_plots, dest_path, color='deepskyblue', fig_title='d)', ax_lim=(0, 800), tick_interval=100)




''' 6. Read and filter GNSS-IR results and calibrate zero-level to date of rover antenna deployment '''

# read and filter gnss-ir snow accumulation results (processed using 'gnssrefl' on Linux)
df_rh = f.read_gnssir(dest_path, reflecto_sol_src_path, base_name, yy_LBLR, copy=False, pickle='nmlb')
gnssir_acc, gnssir_acc_daily, gnssir_acc_daily_std, gnssir_rh_clean = f.filter_gnssir(df_rh, acc_at_time_of_first_obs=Acc_211126, freq='2nd', threshold=2)




''' 7. Calculate accumulation differences, linear regressions & RMSE between GNSS-IR and reference data '''

# calculate and plot linear fit and statistics between GNSS-IR and each reference sensor
GNSSIR_laser_R, GNSSIR_laser_R_p = f.dependency(laser_daily.dsh/10, 'laser [cm]', gnssir_acc_daily.interpolate()/10, 'GNSS-IR [cm]', plot_4_dsh_fit, '4c1_dependence_GNSS-IR_laser', save_plots, dest_path, fig_title='a) Snow Height', color='darkseagreen', unit='cm')
GNSSIR_buoy_R, GNSSIR_buoy_R_p = f.dependency(buoy_daily.dsh_mean/10, 'snow buoy mean [cm]', gnssir_acc_daily.interpolate()/10,  'GNSS-IR [cm]', plot_4_dsh_fit, '4c2_dependence_GNSS-IR_buoy', save_plots, dest_path, fig_title='b) Snow Height', color='darkgray', unit='cm')
GNSSIR_poles_R, GNSSIR_poles_R_p = f.dependency(poles_daily.sh_mean/10, 'stake farm mean [cm]', gnssir_acc_daily.interpolate()/10, 'GNSS-IR [cm]', plot_4_dsh_fit, '4c3_dependence_GNSS-IR_stake', save_plots, dest_path, fig_title='c) Snow Height', color='darkkhaki', unit='cm')




''' 8. Convert GNSS-IR snow height into SWE with density '''

# calculate SWE of GNSS-IR combined with snow pit density and a const mean value
swe_gnssIR_manual = manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].resample('D').interpolate() * gnssir_acc_daily/1000
swe_gnssIR_const = gnssir_acc_daily/1000*408




''' 9. Calculate SWE differences, linear regressions & RMSE between GNSS-refractometry and GNSS-IR SWE reference '''

# GNSS refractometry vs. (GNSS-IR * manual)
SWE_LBLR_gnssirmanual_R, SWE_LBLR_gnssirmanual_R_p, = f.dependency(swe_gnssIR_manual, 'mass (GNSS-IR * snow pit) [kg/m²]', swe_gnss_daily_LBLR.interpolate(), 'mass (LBLR) [kg/m²]', plot_2_SWE_deviation, '2f3a_dependence_SWE_ref2LBLR', save_plots, dest_path, color='crimson', fig_title='a)', ax_lim=(0, 650), tick_interval=100)
SWE_LBUR_gnssirmanual_R, SWE_LBUR_gnssirmanual_R_p, = f.dependency(swe_gnssIR_manual, 'mass (GNSS-IR * snow pit) [kg/m²]', swe_gnss_daily_LBUR.interpolate(), 'mass (LBUR) [kg/m²]', plot_2_SWE_deviation, '2f3b_dependence_SWE_ref2LBUR', save_plots, dest_path, color='salmon', fig_title='c)', ax_lim=(0, 650), tick_interval=100)
SWE_JBLR_gnssirmanual_R, SWE_JBLR_gnssirmanual_R_p, = f.dependency(swe_gnssIR_manual, 'mass (GNSS-IR * snow pit) [kg/m²]', swe_gnss_daily_JBLR.interpolate(), 'mass (JBLR) [kg/m²]', plot_2_SWE_deviation, '2f3c_dependence_SWE_ref2JBLR', save_plots, dest_path, color='dodgerblue', fig_title='b)', ax_lim=(0, 650), tick_interval=100)
SWE_JBUR_gnssirmanual_R, SWE_JBUR_gnssirmanual_R_p, = f.dependency(swe_gnssIR_manual, 'mass (GNSS-IR * snow pit) [kg/m²]', swe_gnss_daily_JBUR.interpolate(), 'mass (JBUR) [kg/m²]', plot_2_SWE_deviation, '2f3d_dependence_SWE_ref2JBUR', save_plots, dest_path, color='deepskyblue', fig_title='d)', ax_lim=(0, 650), tick_interval=100)
# GNSS refractometry vs. (GNSS-IR * const value)
SWE_LBLR_gnssirconst_R, SWE_LBLR_gnssirconst_R_p, = f.dependency(swe_gnssIR_const, 'mass (GNSS-IR * 408 kg/m³) [kg/m²]', swe_gnss_daily_LBLR.interpolate(), 'mass (LBLR) [kg/m²]', plot_2_SWE_deviation, '2f4a_dependence_SWE_ref2LBLR', save_plots, dest_path, color='crimson', fig_title='a)', ax_lim=(0, 650), tick_interval=100)
SWE_LBUR_gnssirconst_R, SWE_LBUR_gnssirconst_R_p, = f.dependency(swe_gnssIR_const, 'mass (GNSS-IR * 408 kg/m³) [kg/m²]', swe_gnss_daily_LBUR.interpolate(), 'mass (LBUR) [kg/m²]', plot_2_SWE_deviation, '2f4b_dependence_SWE_ref2LBUR', save_plots, dest_path, color='salmon', fig_title='c)', ax_lim=(0, 650), tick_interval=100)
SWE_JBLR_gnssirconst_R, SWE_JBLR_gnssirconst_R_p, = f.dependency(swe_gnssIR_const, 'mass (GNSS-IR * 408 kg/m³) [kg/m²]', swe_gnss_daily_JBLR.interpolate(), 'mass (JBLR) [kg/m²]', plot_2_SWE_deviation, '2f4c_dependence_SWE_ref2JBLR', save_plots, dest_path, color='dodgerblue', fig_title='b)', ax_lim=(0, 650), tick_interval=100)
SWE_JBUR_gnssirconst_R, SWE_JBUR_gnssirconst_R_p, = f.dependency(swe_gnssIR_const, 'mass (GNSS-IR * 408 kg/m³) [kg/m²]', swe_gnss_daily_JBUR.interpolate(), 'mass (JBUR) [kg/m²]', plot_2_SWE_deviation, '2f4d_dependence_SWE_ref2JBUR', save_plots, dest_path, color='deepskyblue', fig_title='d)', ax_lim=(0, 650), tick_interval=100)




''' 10. Calculate snow density '''

# calculate from GNSS-RR: Combine GNSS-IR & GNSS refractometry
density_LBLR = f.convert_swesh2density(swe_gnss_daily_LBLR.interpolate(), gnssir_acc_daily.interpolate())
density_LBUR = f.convert_swesh2density(swe_gnss_daily_LBUR.interpolate(), gnssir_acc_daily.interpolate())
density_JBLR = f.convert_swesh2density(swe_gnss_daily_JBLR.interpolate(), gnssir_acc_daily.interpolate())
density_JBUR = f.convert_swesh2density(swe_gnss_daily_JBUR.interpolate(), gnssir_acc_daily.interpolate())

# calculate from GNSS refractometry and SHM laser
density_LBLR_laser = swe_gnss_daily_LBLR.interpolate() / laser_daily.dsh*1000
density_LBUR_laser = swe_gnss_daily_LBUR.interpolate() / laser_daily.dsh*1000
density_JBLR_laser = swe_gnss_daily_JBLR.interpolate() / laser_daily.dsh*1000
density_JBUR_laser = swe_gnss_daily_JBUR.interpolate() / laser_daily.dsh*1000




''' 11. Calculate density differences, linear regressions & RMSE between GNSS-RR and reference data '''

# calculate statistics and plot correlation and fit of...
# GNSS-RR vs. snow pit measurements
GNSSRR_LBLR_manual_R, GNSSRR_LBLR_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_LBLR.interpolate(), 'density GNSS-RR [kg/m³]', plot_6_density_deviation, '6e4a_dependence_GNSS-RR_manual_LBLR', save_plots, dest_path, plot_fit=False, fig_title='a) LBLR', color='crimson', ax_lim=(50, 550), tick_interval=100)
GNSSRR_LBUR_manual_R, GNSSRR_LBUR_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_LBUR.interpolate(), 'density GNSS-RR [kg/m³]', plot_6_density_deviation, '6e4b_dependence_GNSS-RR_manual_LBUR', save_plots, dest_path, plot_fit=False, fig_title='c) LBUR', color='salmon', ax_lim=(50, 550), tick_interval=100)
GNSSRR_JBLR_manual_R, GNSSRR_JBLR_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_JBLR.interpolate(), 'density GNSS-RR [kg/m³]', plot_6_density_deviation, '6e4c_dependence_GNSS-RR_manual_JBLR', save_plots, dest_path, plot_fit=False, fig_title='b) JBLR', color='dodgerblue', ax_lim=(50, 550), tick_interval=100)
GNSSRR_JBUR_manual_R, GNSSRR_JBUR_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_JBUR.interpolate(), 'density GNSS-RR [kg/m³]', plot_6_density_deviation, '6e4d_dependence_GNSS-RR_manual_JBUR', save_plots, dest_path, plot_fit=False, fig_title='d) JBUR', color='deepskyblue', ax_lim=(50, 550), tick_interval=100)
# (GNSS refractometry / laser) vs. snow pit measurements
GNSSRR_LBLR_laser_manual_R, GNSSRR_LBLR_laser_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_LBLR_laser.dropna(), 'density GNSS refractometry/laser [kg/m³]', plot_6_density_deviation, '6e6a_dependence_GNSSrefr_laser_vs_manual_LBLR', save_plots, dest_path, plot_fit=False, fig_title='a) LBLR', color='crimson', ax_lim=(50, 550), tick_interval=100)
GNSSRR_LBUR_laser_manual_R, GNSSRR_LBUR_laser_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_LBUR_laser.dropna(), 'density GNSS refractometry/laser [kg/m³]', plot_6_density_deviation, '6e6b_dependence_GNSSrefr_laser_vs_manual_LBUR', save_plots, dest_path, plot_fit=False, fig_title='c) LBUR', color='salmon', ax_lim=(50, 550), tick_interval=100)
GNSSRR_JBLR_laser_manual_R, GNSSRR_JBLR_laser_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_JBLR_laser.dropna(), 'density GNSS refractometry/laser [kg/m³]', plot_6_density_deviation, '6e6c_dependence_GNSSrefr_laser_vs_manual_JBLR', save_plots, dest_path, plot_fit=False, fig_title='b) JBLR', color='dodgerblue', ax_lim=(50, 550), tick_interval=100)
GNSSRR_JBUR_laser_manual_R, GNSSRR_JBUR_laser_manual_R_p = f.dependency(manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'], 'density snow pit [kg/m³]', density_JBUR_laser.dropna(), 'density GNSS refractometry/laser [kg/m³]', plot_6_density_deviation, '6e6d_dependence_GNSSrefr_laser_vs_manual_JBUR', save_plots, dest_path, plot_fit=False, fig_title='d) JBUR', color='deepskyblue', ax_lim=(50, 550), tick_interval=100)




''' 12. Exponential regression analysis of SWE versus snow height and derivative for density versus snow height  '''

# exponential regression - curve fitting by minimizing the sum of squared residuals

# SWE versus snow height
# calculate (y-values of) exponential regression curve, print function parameters and statistics
swe_over_sh_datetime, swe_over_sh = f.create_new_df(gnssir_acc_daily.interpolate()/1000, 'GNSS-IR_sh', swe_gnss_daily_LBLR.interpolate(), 'swe_gnss_daily_LBLR', swe_gnss_daily_LBUR.interpolate(), 'swe_gnss_daily_LBUR', swe_gnss_daily_JBLR.interpolate(), 'swe_gnss_daily_JBLR', swe_gnss_daily_JBUR.interpolate(), 'swe_gnss_daily_JBUR')
swe_over_sh.loc[0] = [0, 0, 0, 0]  # adding a row
swe_over_sh = swe_over_sh.sort_index()  # sorting by index
exp_fit_LBLR = f.exponential_regression('swe versus snow height - LBLR', np.array(swe_over_sh.index), np.array(swe_over_sh.swe_gnss_daily_LBLR.tolist()), y_0=0)
exp_fit_LBUR = f.exponential_regression('swe versus snow height - LBUR', np.array(swe_over_sh.index), np.array(swe_over_sh.swe_gnss_daily_LBUR.tolist()), y_0=0)
exp_fit_JBLR = f.exponential_regression('swe versus snow height - JBLR', np.array(swe_over_sh.index), np.array(swe_over_sh.swe_gnss_daily_JBLR.tolist()), y_0=0)
exp_fit_JBUR = f.exponential_regression('swe versus snow height - JBUR', np.array(swe_over_sh.index), np.array(swe_over_sh.swe_gnss_daily_JBUR.tolist()), y_0=0)
# create out of the exponential fit function a vector that contains in 0.01 spacing all x-values
exp_fit_LBLR = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_LBLR[2][0], exp_fit_LBLR[2][1], exp_fit_LBLR[2][2]), exp_fit_LBLR[2]]
exp_fit_LBUR = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_LBUR[2][0], exp_fit_LBUR[2][1], exp_fit_LBUR[2][2]), exp_fit_LBUR[2]]
exp_fit_JBLR = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_JBLR[2][0], exp_fit_JBLR[2][1], exp_fit_JBLR[2][2]), exp_fit_JBLR[2]]
exp_fit_JBUR = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_JBUR[2][0], exp_fit_JBUR[2][1], exp_fit_JBUR[2][2]), exp_fit_JBUR[2]]
# same from 01.03.2022 onwards
swe_over_sh_datetime_0103, swe_over_sh_0103 = f.create_new_df(gnssir_acc_daily[gnssir_acc_daily.index >= '2022-03-01'].interpolate()/1000, 'GNSS-IR_sh', swe_gnss_daily_LBLR.interpolate(), 'swe_gnss_daily_LBLR', swe_gnss_daily_LBUR.interpolate(), 'swe_gnss_daily_LBUR', swe_gnss_daily_JBLR.interpolate(), 'swe_gnss_daily_JBLR', swe_gnss_daily_JBUR.interpolate(), 'swe_gnss_daily_JBUR')
swe_over_sh_0103.loc[0] = [0, 0, 0, 0]  # adding a row
swe_over_sh_0103 = swe_over_sh_0103.sort_index()  # sorting by index
exp_fit_LBLR_0103 = f.exponential_regression('swe versus snow (starting 1/3/22) - LBLR', np.array(swe_over_sh_0103.index), np.array(swe_over_sh_0103.swe_gnss_daily_LBLR.tolist()), y_0=0)
exp_fit_LBUR_0103 = f.exponential_regression('swe versus snow height (starting 1/3/22) - LBUR', np.array(swe_over_sh_0103.index), np.array(swe_over_sh_0103.swe_gnss_daily_LBUR.tolist()), y_0=0)
exp_fit_JBLR_0103 = f.exponential_regression('swe versus snow height (starting 1/3/22) - JBLR', np.array(swe_over_sh_0103.index), np.array(swe_over_sh_0103.swe_gnss_daily_JBLR.tolist()), y_0=0)
exp_fit_JBUR_0103 = f.exponential_regression('swe versus snow height (starting 1/3/22) - JBUR', np.array(swe_over_sh_0103.index), np.array(swe_over_sh_0103.swe_gnss_daily_JBUR.tolist()), y_0=0)
# create out of the exponential fit function a vector that contains in 0.01 spacing all x-values
exp_fit_LBLR_0103 = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_LBLR_0103[2][0], exp_fit_LBLR_0103[2][1], exp_fit_LBLR_0103[2][2]), exp_fit_LBLR_0103[2]]
exp_fit_LBUR_0103 = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_LBUR_0103[2][0], exp_fit_LBUR_0103[2][1], exp_fit_LBUR_0103[2][2]), exp_fit_LBUR_0103[2]]
exp_fit_JBLR_0103 = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_JBLR_0103[2][0], exp_fit_JBLR_0103[2][1], exp_fit_JBLR_0103[2][2]), exp_fit_JBLR_0103[2]]
exp_fit_JBUR_0103 = [np.linspace(0, 1.5, 151), f.func_exp(np.linspace(0, 1.5, 151), exp_fit_JBUR_0103[2][0], exp_fit_JBUR_0103[2][1], exp_fit_JBUR_0103[2][2]), exp_fit_JBUR_0103[2]]

# density versus snow height
# create a new dataframe for plotting density versus snow height
density_over_sh_datetime, density_over_sh = f.create_new_df(gnssir_acc_daily.interpolate()/1000, 'GNSS-IR_sh', density_LBLR.interpolate(), 'density_LBLR', y_data2=density_LBUR.interpolate(), y_data2_name='density_LBUR', y_data3=density_JBLR.interpolate(), y_data3_name='density_JBLR', y_data4=density_JBUR.interpolate(), y_data4_name='density_JBUR')
density_over_sh_datetime_0103, density_over_sh_0103 = f.create_new_df(gnssir_acc_daily[gnssir_acc_daily.index >= '2022-03-01'].interpolate()/1000, 'GNSS-IR_sh', density_LBLR.interpolate(), 'density_LBLR', y_data2=density_LBUR.interpolate(), y_data2_name='density_LBUR', y_data3=density_JBLR.interpolate(), y_data3_name='density_JBLR', y_data4=density_JBUR.interpolate(), y_data4_name='density_JBUR')
# calculate the fit for density versus snow height by calculating the trend in each point from zero (delta mass / delta snow height)
fit_sh_density_LBLR = [exp_fit_LBLR[0], exp_fit_LBLR[1]/exp_fit_LBLR[0]]
fit_sh_density_LBUR = [exp_fit_LBUR[0], exp_fit_LBUR[1]/exp_fit_LBUR[0]]
fit_sh_density_JBLR = [exp_fit_JBLR[0], exp_fit_JBLR[1]/exp_fit_JBLR[0]]
fit_sh_density_JBUR = [exp_fit_JBUR[0], exp_fit_JBUR[1]/exp_fit_JBUR[0]]
# from 01.03.2022 onwards
fit_sh_density_LBLR_0103 = [exp_fit_LBLR_0103[0], exp_fit_LBLR_0103[1]/exp_fit_LBLR_0103[0]]
fit_sh_density_LBUR_0103 = [exp_fit_LBUR_0103[0], exp_fit_LBUR_0103[1]/exp_fit_LBUR_0103[0]]
fit_sh_density_JBLR_0103 = [exp_fit_JBLR_0103[0], exp_fit_JBLR_0103[1]/exp_fit_JBLR_0103[0]]
fit_sh_density_JBUR_0103 = [exp_fit_JBUR_0103[0], exp_fit_JBUR_0103[1]/exp_fit_JBUR_0103[0]]
# calculate the fit curve for density by the analytical derivative of fit swe versus snow height
fit_derivative_LBLR = [exp_fit_LBLR[0], f.func_exp2(exp_fit_LBLR[0], 128.1, 1.22)]
fit_derivative_LBUR = [exp_fit_LBUR[0], f.func_exp2(exp_fit_LBUR[0], 125.24, 1.24)]
fit_derivative_JBLR = [exp_fit_JBLR[0], f.func_exp2(exp_fit_JBLR[0], 152.88, 0.84)]
fit_derivative_JBUR = [exp_fit_JBUR[0], f.func_exp2(exp_fit_JBUR[0], 175.26, 0.69)]
# from 01.03.2022 onwards
fit_derivative_LBLR_0103 = [exp_fit_LBLR_0103[0], f.func_exp2(exp_fit_LBLR_0103[0], 130.8, 1.20)]
fit_derivative_LBUR_0103 = [exp_fit_LBUR_0103[0], f.func_exp2(exp_fit_LBUR_0103[0], 119.6, 1.30)]
fit_derivative_JBLR_0103 = [exp_fit_JBLR_0103[0], f.func_exp2(exp_fit_JBLR_0103[0], 148.5, 0.90)]
fit_derivative_JBUR_0103 = [exp_fit_JBUR_0103[0], f.func_exp2(exp_fit_JBUR_0103[0], 162.36, 0.82)]




''' 13. Calculate GNSS-RR density uncertainty over snow height by error propagation '''

# Insert different values for the uncertainty of GNSS refractometry SWE [kg/m²] and GNSS-IR snow height [m]
swe_err = [10, 30, 50, 90]
h_err = [0.1, 0.2, 0.3, 0.4, 0.25]

# evolution of relative and absolute density error  over snow height inserting exponential regression parameters
density_err_h01 = f.func_err_prop(exp_fit_LBLR[2][0], exp_fit_LBLR[2][1], exp_fit_LBLR[2][2], swe_err[0], swe_err[1], swe_err[2], swe_err[3], h_err[0], h_err[0], h_err[0], h_err[0])
density_err_m30 = f.func_err_prop(exp_fit_LBLR[2][0], exp_fit_LBLR[2][1], exp_fit_LBLR[2][2], swe_err[1], swe_err[1], swe_err[1], swe_err[1], h_err[0], h_err[1], h_err[2], h_err[3])

# calculate error for constant densities over snow height
density = [50, 150, 300, 550]
density_err = f.func_err_prop2(exp_fit_LBLR[2][0], exp_fit_LBLR[2][1], exp_fit_LBLR[2][2], swe_err[1], h_err[4], density[0], density[1], density[2], density[3])




''' 14. Calculate height, mass and density of a freshly fallen snow layer '''

# properties of new snow layer that accumulated over a specified interval and comprising a minimum height
new_snow_datetimeindex_LBLR, new_snow_heightindex_LBLR = f.calc_new_snow_density(gnssir_acc_daily/1000, swe_gnss_daily_LBLR.interpolate(), interval=new_snow[5][0], min_acc=new_snow[5][1])
new_snow_datetimeindex_JBLR, new_snow_heightindex_JBLR = f.calc_new_snow_density(gnssir_acc_daily/1000, swe_gnss_daily_JBLR.interpolate(), interval=new_snow[5][0], min_acc=new_snow[5][1])




''' 15. Plotting '''

# Directory for saving all figures
os.makedirs(dest_path + '30_plots/', exist_ok=True)


''' 15. 0) GNSS refractometry solution quality '''

# 0a) Number of satellites
f.plot_ds(dest_path, '0a_nb_satellites_LR',
          create_plot=plot_0_sol_quality,
          fig_size=(12, 5.5),
          legend_position=(0.01, 0.22),
          save=save_plots,
          y_label='Number of Satellites',
          ytick_interval=2,
          y_lim=(1, 15),
          x_lim=xlim_dates,
          ds1=df_enu_LBLR.nr_sat,
          ds1_transparency=0.3,
          ds3=df_enu_JBLR.nr_sat,
          ds3_transparency=0.5)
f.plot_ds(dest_path, '0a_nb_satellites_ER',
          create_plot=plot_0_sol_quality,
          fig_size=(12, 5.5),
          legend_position=(0.01, 0.22),
          save=save_plots,
          y_label='Number of Satellites',
          ytick_interval=2,
          y_lim=(1, 15),
          x_lim=xlim_dates,
          ds2=fil_df_LBUR.nr_sat,
          ds2_transparency=0.2,
          ds4=fil_df_JBUR.nr_sat,
          ds4_transparency=0.6)

# 0b) Ambiguity resolution state
f.plot_solquality(dest_path, df_enu_LBLR.amb_state, df_enu_LBUR.amb_state, create_plot=plot_0_sol_quality, save=save_plots, suffix='LB', y_lim=(0, 100), x_lim=xlim_dates)
f.plot_solquality(dest_path, df_enu_JBLR.amb_state, df_enu_JBUR.amb_state, create_plot=plot_0_sol_quality, save=save_plots, suffix='JB', y_lim=(0, 100), x_lim=xlim_dates)

# 0c) Daily noise versus number of fixed ambiguities for each baseline
f.plot_ds(dest_path,
          '0c_1_Solutions_Quality_Control_LBLR',
          create_plot=plot_0_sol_quality,
          save=save_plots,
          fig_title='a)',
          fig_size=(7, 5),
          legend_position=(0.38, 0.83),
          y_lim=(0, 75),
          ytick_interval=10,
          x_datetime=False,
          x_lim=(0, 100),
          xtick_interval=10,
          y_label='Daily Noise [kg/m²]',
          x_label='Number of Fixed Ambiguities per Day',
          ds1_fit=amb_std_fit_LBLR,
          ds1_fit_label="fit: {:.1f} / sqrt(n)".format(*amb_std_fit_LBLR[2]),
          ds1=amb_std_LBLR,
          ds1_label='LBLR',
          ds1_linestyle='',
          ds1_marker='o',
          ds7_fit=func_exp_LBLR,
          ds7_fit_label="func = " + str(func_exp_LBLR[2]) + " / sqrt(n)")
f.plot_ds(dest_path,
          '0c_2_Solutions_Quality_Control_LBUR',
          create_plot=plot_0_sol_quality,
          save=save_plots,
          fig_title='c)',
          fig_size=(7, 5),
          legend_position=(0.38, 0.83),
          y_lim=(0, 75),
          ytick_interval=10,
          x_datetime=False,
          x_lim=(0, 100),
          xtick_interval=10,
          y_label='Daily Noise [kg/m²]',
          x_label='Number of Fixed Ambiguities per Day',
          ds2_fit=amb_std_fit_LBUR,
          ds2_fit_label="fit: {:.1f} / sqrt(n)".format(*amb_std_fit_LBUR[2]),
          ds2=amb_std_LBUR,
          ds2_label='LBUR',
          ds2_linestyle='',
          ds2_marker='o',
          ds7_fit=func_exp_LBUR,
          ds7_fit_label="func = " + str(func_exp_LBUR[2]) + " / sqrt(n)")
f.plot_ds(dest_path,
          '0c_3_Solutions_Quality_Control_JBLR',
          create_plot=plot_0_sol_quality,
          save=save_plots,
          fig_title='b)',
          fig_size=(7, 5),
          legend_position=(0.38, 0.83),
          y_lim=(0, 75),
          ytick_interval=10,
          x_datetime=False,
          x_lim=(0, 100),
          xtick_interval=10,
          y_label='Daily Noise [kg/m²]',
          x_label='Number of Fixed Ambiguities per Day',
          ds3_fit=amb_std_fit_JBLR,
          ds3_fit_label="fit: {:.1f} / sqrt(n)".format(*amb_std_fit_JBLR[2]),
          ds3=amb_std_JBLR,
          ds3_label='JBLR',
          ds3_linestyle='',
          ds3_marker='o',
          ds7_fit=func_exp_JBLR,
          ds7_fit_label="func = " + str(func_exp_JBLR[2]) + " / sqrt(n)")
f.plot_ds(dest_path,
          '0c_4_Solutions_Quality_Control_JBUR',
          create_plot=plot_0_sol_quality,
          save=save_plots,
          fig_title='d)',
          fig_size=(7, 5),
          legend_position=(0.38, 0.83),
          y_lim=(0, 75),
          ytick_interval=10,
          x_datetime=False,
          x_lim=(0, 100),
          xtick_interval=10,
          y_label='Daily Noise [kg/m²]',
          x_label='Number of Fixed Ambiguities per Day',
          ds4_fit=amb_std_fit_JBUR,
          ds4_fit_label="fit: {:.1f} / sqrt(n)".format(*amb_std_fit_JBUR[2]),
          ds4=amb_std_JBUR,
          ds4_label='JBUR',
          ds4_linestyle='',
          ds4_marker='o',
          ds7_fit=func_exp_JBUR,
          ds7_fit_label="func = " + str(func_exp_JBUR[2]) + " / sqrt(n)")


''' 15. 1) Show each filtering step for all baselines '''

#  a) Before filtering - "raw" ENU solutions
f.plot_ds(dest_path,
          '1a_ALL_ENU',
          fig_title='Baseline Up-Component (U) unfiltered "df_enu"',
          create_plot=plot_1_filtering,
          save=save_plots,
          y_label='U [m]',
          ytick_interval=2,
          y_lim=(-8, 14),
          ds1=df_enu_LBLR.U.dropna(),
          ds2=df_enu_LBUR.U.dropna(),
          ds3=df_enu_JBLR.U.dropna(),
          ds4=df_enu_JBUR.U.dropna())

# b) after selecting only fixed ambiguities:
f.plot_ds(dest_path,
          '1b_ALL_fil_df_Am',
          fig_title='Solutions with fixed ambiguities "fil_df"',
          create_plot=plot_1_filtering,
          save=save_plots,
          y_label='U [m]',
          ytick_interval=2,
          y_lim=(-8, 14),
          ds1=fil_df_LBLR.U.dropna(),
          ds2=fil_df_LBUR.U.dropna(),
          ds3=fil_df_JBLR.U.dropna(),
          ds4=fil_df_JBUR.U.dropna())

# c) after translating values to [mm] (*1000) and correcting for SMH-events:
f.plot_ds(dest_path,
          '1c_ALL_U_Am_SMH',
          fig_title='Corrected for Snow-Mast-Heightening events "u"',
          create_plot=plot_1_filtering,
          save=save_plots,
          y_label='U [mm]',
          ytick_interval=200,
          y_lim=(-3400, -1400),
          legend_position=(0.01, 0.53),
          ds1=u_LBLR.dropna(),
          ds2=u_LBUR.dropna(),
          ds3=u_JBLR.dropna(),
          ds4=u_JBUR.dropna())

# d) after removing outliers based on 3*sigma threshold
f.plot_ds(dest_path,
          '1d_ALL_U_clean_Am_SMH_3sLR_2sER',
          fig_title='Outliers removed "u_clean"',
          create_plot=plot_1_filtering,
          save=save_plots,
          y_label='U [mm]',
          ytick_interval=200,
          y_lim=(-3400, -1400),
          legend_position=(0.01, 0.53),
          ds1=u_clean_LBLR.dropna(),
          ds2=u_clean_LBUR.dropna(),
          ds3=u_clean_JBLR.dropna(),
          ds4=u_clean_JBUR.dropna())

# e) after correcting values to be positive values
f.plot_ds(dest_path,
          '1e_ALL_swe_unfil_Am_SMH_3sLR_2sER_2SWE',
          fig_title='Up-component subtracted by physical baseline length "swe_unfil"',
          create_plot=plot_1_filtering,
          save=save_plots,
          y_lim=(-50, 800),
          ds1=swe_unfil_LBLR.dropna(),
          ds2=swe_unfil_LBUR.dropna(),
          ds3=swe_unfil_JBLR.dropna(),
          ds4=swe_unfil_JBUR.dropna())

# f) after filtering data with a rolling median
f.plot_ds(dest_path,
          '1f_ALL_swe_gnss_Am_SMH_3sLR_2sER_2SWE_median',
          fig_title='SWE filtered by rolling median (1 day) "swe_gnss"',
          create_plot=plot_1_filtering,
          save=save_plots,
          y_lim=(-50, 800),
          ds1=swe_gnss_LBLR.dropna(),
          ds1_std=std_gnss_LBLR.dropna(),
          ds2=swe_gnss_LBUR.dropna(),
          ds2_std=std_gnss_LBUR.dropna(),
          ds3=swe_gnss_JBLR.dropna(),
          ds3_std=std_gnss_JBLR.dropna(),
          ds4=swe_gnss_JBUR.dropna(),
          ds4_std=std_gnss_JBUR.dropna())


''' 15. 2) SWE of GNSS refractometry and reference '''

# 2a) GNSS refractometry SWE of all baselines
f.plot_ds(dest_path,
          '2a_SWE_GNSS',
          fig_title='GNSS Snow Water Equivalent Measurements "swe_gnss_daily"',
          create_plot=plot_2_SWE_GNSS,
          save=save_plots,
          y_label='mass [kg/m²]',
          y_lim=(-50, 800),
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds2=swe_gnss_daily_LBUR.dropna(),
          ds2_std=std_gnss_daily_LBUR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds4=swe_gnss_daily_JBUR.dropna(),
          ds4_std=std_gnss_daily_JBUR.dropna())

# 2b) high-end GNSS refractometry SWE and reference SWE
f.plot_ds(dest_path, '2b_SWE_GNSS_highend',
          # fig_title='High-End GNSS "swe_gnss_daily"',
          create_plot=plot_2_SWE_GNSS,
          save=save_plots,
          legend_position=(0.01, 0.99),
          x_lim=(dt.date(2021, 12, 1), dt.date(2023, 8, 15)),
          y_lim=(-99, 800),
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_linestyle='',
          ds1_label=r'$\bf{GNSS refractometry}$',
          ds2=swe_gnss_daily_LBLR.dropna(),
          ds2_std=std_gnss_daily_LBLR.dropna(),
          ds2_label='Leica Base to Leica Rover (LBLR)',
          ds2_color='crimson',
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds5=laser_daily.dswe,
          ds5_marker='',
          ds5_label=r'$\bf{Reference Measurements}$',
          ds6=laser_daily.dswe,
          ds6_bias=laser_daily.dswe*-0.25,
          ds6_std_label='bias',
          ds6_label='laser h * snow pit ρ',
          ds6_color='forestgreen',
          ds7_label='laser h * 408 kg/m³',
          ds7=laser_daily.dswe_const,
          ds7_color='darkseagreen',
          ds8=swe_gnssIR_manual,
          ds8_label='GNSS-IR h * snow pit ρ',
          ds8_color='chocolate',
          ds8_linestyle='-',
          ds8_bias=swe_gnssIR_manual * -0.25,
          ds8_std_label='bias',
          ds9=swe_gnssIR_const,
          ds9_yaxis=1,
          ds9_label='GNSS-IR h * 408 kg/m³')

# 2b) SWE: CUT to beginning period (first months)
f.plot_ds(dest_path, '2b2_SWE_GNSS_highend_cut',
          # fig_title='High-End GNSS "swe_gnss_daily"',
          create_plot=plot_2_SWE_GNSS,
          save=save_plots,
          legend_position=(0.01, 0.99),
          x_lim=(dt.date(2021, 11, 15), dt.date(2022, 3, 5)),
          y_lim=(-50, 300),
          ytick_interval=50,
          x_locator='day',
          y_axis2=True,
          legend_title='mass:',
          legend2_title='snow height:',
          legend2_position=(0.8, 0.99),
          y2_lim=(-12.5, 74.2),
          y2tick_interval=10,
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6=laser_daily.dswe,
          ds6_bias=laser_daily.dswe*-0.25,
          ds6_std_label='bias',
          ds6_label='laser h * snow pit ρ',
          ds6_color='forestgreen',
          ds7_label='laser h * 408 kg/m³',
          ds7=laser_daily.dswe_const,
          ds7_color='darkseagreen',
          ds7_std_label='bias',
          ds2=laser_daily.dsh* -1000,
          ds2_yaxis=2,
          ds2_label='laser',
          ds2_linestyle='',
          ds2_marker='o',
          ds2_color='darkseagreen',
          ds4=swe_gnssIR_manual,
          ds4_bias=swe_gnssIR_manual * -0.25,
          ds4_std_label='bias',
          ds4_label='GNSS-IR h * snow pit ρ',
          ds4_color='chocolate',
          ds4_linestyle='-',
          ds5=swe_gnssIR_const,
          ds5_label='GNSS-IR h * 408 kg/m³',
          ds5_linestyle='-',
          ds5_color='goldenrod',
          ds5_marker='',
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10)

# 2c) Low-cost GNSS refractometry and reference SWE
f.plot_ds(dest_path,
          '2c_SWE_GNSS_lowcost',
          create_plot=plot_2_SWE_GNSS,
          save=save_plots,
          legend_position=(0.01, 1),
          x_lim=(dt.date(2021, 12, 1), dt.date(2023, 8, 15)),
          y_lim=(-99, 800),
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_linestyle='',
          ds1_label=r'$\bf{GNSS refractometry}$',
          ds2=swe_gnss_daily_LBUR.dropna(),
          ds2_std=std_gnss_daily_LBUR.dropna(),
          ds4=swe_gnss_daily_JBUR.dropna(),
          ds4_std=std_gnss_daily_JBUR.dropna(),
          ds5=laser_daily.dswe,
          ds5_marker='',
          ds5_label=r'$\bf{Reference Measurements}$',
          ds6=laser_daily.dswe,
          ds6_label='laser sh * interpolated snow pit ρ',
          ds6_color='forestgreen',
          ds6_bias=laser_daily.dswe * -0.25,
          ds6_std_label='bias',
          ds7_label='laser sh * 408 kg/m³',
          ds7=laser_daily.dswe_const,
          ds7_color='darkseagreen',
          ds8=swe_gnssIR_manual,
          ds8_label='GNSS-IR h * snow pit ρ',
          ds8_color='chocolate',
          ds8_linestyle='-',
          ds8_bias=swe_gnssIR_manual * -0.25,
          ds8_std_label='bias',
          ds9=swe_gnssIR_const,
          ds9_yaxis=1,
          ds9_label='GNSS-IR h * 408 kg/m³')

# 2e) Deviation of different baseline solutions
# 2e) 1 high-end solutions
f.plot_ds(dest_path,
          '2e1a_deviation_JB2LB_highend_solutions_absolut_cut',
          fig_title='a) High-end rover solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          fig_size=(15, 6),    # CUT
          y_lim=(-70, 140),    # CUT
          #y_lim=(-100, 300),    # same size like the other Figures
          ytick_interval=20,
          y_label='∆m (LBLR-JBLR) [kg/m²]',
          ds1=deviation_JB2LB_he,
          ds1_color='purple',
          ds1_label=None,
          plot_vline=dt.datetime(2022, 2, 15),
          hline_value=deviation_JB2LB_he[deviation_JB2LB_he.index >= '2022-02-15'].mean())
f.plot_ds(dest_path,
          '2e1b_deviation_JB2LB_highend_solutions_prozent',
          fig_title='b) High-end rover solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-100, 300),
          y_label='∆m (LBLR-JBLR) [%]',
          ds1=deviation_p_JB2LB_he,
          ds1_color='purple',
          ds1_label=None,
          plot_vline=dt.datetime(2022, 2, 15))
# 2e) 2 Leica base solutions
f.plot_ds(dest_path,
          '2e2a_deviation_UR2LR_LB_solutions_absolut',
          fig_title='a) Leica base solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-140, 140),
          y_label='∆m (LBLR-LBUR) [kg/m²]',
          ds1=deviation_UR2LR_LB,
          ds1_color='indianred',
          ds1_label=None)
f.plot_ds(dest_path,
          '2e2b_deviation_UR2LR_LB_solutions_prozent',
          fig_title='b) Leica base solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-100, 300),
          y_label='∆m (LBLR-LBUR) [%]',
          ds1=deviation_p_UR2LR_LB,
          ds1_color='indianred',
          ds1_label=None,
          plot_vline=dt.datetime(2022, 7, 1),
          hline_value=deviation_p_UR2LR_LB[deviation_p_UR2LR_LB.index >= '2022-07-01'].mean())
# 2e) 3 JAVAD base solutions
f.plot_ds(dest_path,
          '2e3a_deviation_UR2LR_JB_solutions_absolut',
          fig_title='a) JAVAD base solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-140, 140),
          y_label='∆m (JBLR-JBUR) [kg/m²]',
          ds1=deviation_UR2LR_JB,
          ds1_color='cornflowerblue',
          ds1_label=None)
f.plot_ds(dest_path,
          '2e3b_deviation_UR2LR_JB_solutions_prozent',
          fig_title='b) JAVAD base solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-100, 300),
          y_label='∆m (JBLR-JBUR) [%]',
          ds1=deviation_p_UR2LR_JB,
          ds1_color='cornflowerblue',
          ds1_label=None,
          plot_vline=dt.datetime(2022, 7, 1),
          hline_value=deviation_p_UR2LR_JB[deviation_p_UR2LR_JB.index >= '2022-07-01'].mean())
# 2e) 4 low-cost solutions
f.plot_ds(dest_path,
          '2e4a_deviation_JB2LB_lowcost_solutions_absolut',
          fig_title='a) Low-cost rover solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-140, 140),
          y_label='∆m (LBUR-JBUR) [kg/m²]',
          ds1=deviation_JB2LB_lc,
          ds1_color='chocolate',
          ds1_label=None)
f.plot_ds(dest_path,
          '2e4b_deviation_JB2LB_lowcost_solutions_prozent',
          fig_title='b) Low-cost rover solution comparison',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          ytick_interval=20,
          y_lim=(-100, 300),
          y_label='∆m (LBUR-JBUR) [%]',
          ds1=deviation_p_JB2LB_lc,
          ds1_color='chocolate',
          ds1_label=None)

# 2f) 1 Deviation between GNSS refractometry and reference sensors SWE (laser * snow pit)
f.plot_ds(dest_path,
          '2f1e_deviation_SWE_GNSS_ref',
          fig_title='a) Deviation: Reference SWE (laser * snow pit) - GNSS Refractometry',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-380, 305),
          ytick_interval=50,
          y_label='∆m (reference - GNSS refractometry) [kg/m²]',
          ds1=SWE_LBLR_lasermanual_R.dropna(),
          ds1_linestyle=' ',
          ds1_marker='o',
          ds2=SWE_LBUR_lasermanual_R.dropna(),
          ds2_linestyle=' ',
          ds2_marker='o',
          ds3=SWE_JBLR_lasermanual_R.dropna(),
          ds3_linestyle=' ',
          ds3_marker='o',
          ds4=SWE_JBUR_lasermanual_R.dropna(),
          ds4_linestyle=' ',
          ds4_marker='o',
          hline_value=0)
# 2f) 2 Deviation between GNSS refractometry and reference sensors SWE (laser * const density)
f.plot_ds(dest_path,
          '2f2e_deviation_SWE_GNSS_ref',
          fig_title='b) Deviation: Reference SWE (laser * 408 kg/m³) - GNSS Refractometry',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-380, 305),
          ytick_interval=50,
          y_label='∆m (reference - GNSS refractometry) [kg/m²]',
          ds1=SWE_LBLR_laserconst_R.dropna(),
          ds1_linestyle=' ',
          ds1_marker='o',
          ds2=SWE_LBUR_laserconst_R.dropna(),
          ds2_linestyle=' ',
          ds2_marker='o',
          ds3=SWE_JBLR_laserconst_R.dropna(),
          ds3_linestyle=' ',
          ds3_marker='o',
          ds4=SWE_JBUR_laserconst_R.dropna(),
          ds4_linestyle=' ',
          ds4_marker='o',
          hline_value=0)
# 2f) 3 Deviation between GNSS refractometry and reference sensors SWE (GNSS-IR * snow pit)
f.plot_ds(dest_path,
          '2f3e_deviation_SWE_GNSS_ref',
          fig_title='a) Deviation: Reference SWE (GNSS-IR * snow pit) -  GNSS Refractometry',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-380, 305),
          ytick_interval=50,
          y_label='∆m (reference - GNSS refractometry) [kg/m²]',
          ds1=SWE_LBLR_gnssirmanual_R.dropna(),
          ds1_linestyle=' ',
          ds1_marker='o',
          ds2=SWE_LBUR_gnssirmanual_R.dropna(),
          ds2_linestyle=' ',
          ds2_marker='o',
          ds3=SWE_JBLR_gnssirmanual_R.dropna(),
          ds3_linestyle=' ',
          ds3_marker='o',
          ds4=SWE_JBUR_gnssirmanual_R.dropna(),
          ds4_linestyle=' ',
          ds4_marker='o',
          hline_value=0)
# 2f) 4 Deviation between GNSS refractometry and reference sensors SWE (GNSS-IR * const density)
f.plot_ds(dest_path,
          '2f4e_deviation_SWE_GNSS_ref',
          fig_title='b) Deviation: Reference SWE (GNSS-IR * 408 kg/m³) - GNSS Refractometry',
          create_plot=plot_2_SWE_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-380, 305),
          ytick_interval=50,
          y_label='∆m (reference - GNSS refractometry) [kg/m²]',
          ds1=SWE_LBLR_gnssirconst_R.dropna(),
          ds1_linestyle=' ',
          ds1_marker='o',
          ds2=SWE_LBUR_gnssirconst_R.dropna(),
          ds2_linestyle=' ',
          ds2_marker='o',
          ds3=SWE_JBLR_gnssirconst_R.dropna(),
          ds3_linestyle=' ',
          ds3_marker='o',
          ds4=SWE_JBUR_gnssirconst_R.dropna(),
          ds4_linestyle=' ',
          ds4_marker='o',
          hline_value=0)


''' 15. 3) Comparison with meteorologic data '''

# 3a) 1 Temperature and sunshine indicator
f.plot_ds(dest_path, '3a1_temperature',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(15, 5),
          y_axis2=True,
          x_lim=xlim_dates,
          y2_lim=(0.1, 1),
          ytick_interval=10,
          y_label='2m air temperature [°C]',
          y2_label=' ',
          ds4=mob['2m Level Temperature'],
          ds4_color='firebrick',
          ds4_label=None,
          ds4_transparency=0,
          ds3=mob['Sunshine Indicator'],
          ds3_color='orange',
          ds3_label=None,
          ds3_transparency=0.65,
          ds3_yaxis=2)
# 3a) 2 Sunshine and SWE
f.plot_ds(dest_path, '3a2_temperature',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(15, 5),
          legend_position=(0.01, 0.98),
          legend2_position=(0.75, 0.2),
          y_axis2=True,
          x_lim=xlim_dates,
          y2_lim=(0.1, 1),
          y2_label=' ',
          y_lim=(0, 700),
          ytick_interval=100,
          ds1=swe_gnss_LBLR.dropna(),
          ds1_std=std_gnss_LBLR.dropna(),
          ds3=swe_gnss_JBLR.dropna(),
          ds3_std=std_gnss_JBLR.dropna(),
          ds4=mob['Sunshine Indicator'],
          ds4_color='orange',
          ds4_label='Sunshine Indicator',
          ds4_transparency=0.65,
          ds4_yaxis=2)
# 3a) 3 Sunshine and density
f.plot_ds(dest_path, '3a3_temperature',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(15, 5),
          legend_position=(0.01, 0.98),
          legend2_position=(0.75, 0.2),
          y_axis2=True,
          x_lim=xlim_dates,
          y2_lim=(0.1, 1),
          y2_label=' ',
          y_lim=(70, 450),
          ytick_interval=50,
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds2=density_LBLR_laser,
          ds2_label='LBLR / laser',
          ds2_color='darkseagreen',
          ds4=density_JBLR_laser,
          ds4_label='JBLR / laser',
          ds4_color='forestgreen',
          ds9=mob['Sunshine Indicator'],
          ds9_color='orange',
          ds9_label='Sunshine Indicator',
          ds9_transparency=0.65)


# 3b) SWE of high-end, wind directionand wind speed in 3 figures
f.plot_ds(dest_path, '3b1_SWE_GNSS',
          fig_title='            SWE of high-end GNSS',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(15, 5),
          legend_position=(0.01, 0.99),
          x_lim=(dt.date(2022, 8, 11), dt.date(2022, 10, 5)),
          y_lim=(290, 370),
          ytick_interval=20,
          ds1=swe_gnss_LBLR.dropna(),
          ds1_std=std_gnss_LBLR.dropna(),
          ds3=swe_gnss_JBLR.dropna(),
          ds3_std=std_gnss_JBLR.dropna(),
          x_locator='day',
          major_day_locator=5)
f.plot_ds(dest_path, '3b2_windspeed',
          fig_title='2 m Level Wind Speed',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(15, 5),
          x_lim=(dt.date(2022, 8, 11), dt.date(2022, 10, 5)),
          y_lim=(-1, 35),
          ytick_interval=5,
          y_label='wind speed [m/s]',
          ds2=mob['2m Level Wind Speed'],
          ds2_label=None,
          ds2_color='darkgreen',
          ds2_transparency=0.65,
          x_locator='day',
          major_day_locator=5)
f.plot_ds(dest_path, '3b3_winddirection',
          fig_title='2 m Level Wind Direction',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(15, 5),
          x_lim=(dt.date(2022, 8, 11), dt.date(2022, 10, 5)),
          y_lim=(0, 360),
          ytick_interval=45,
          y_label='wind direction [°]',
          ds2=mob['2m Level Wind Direction'],
          ds2_color='darkgreen',
          ds2_label=None,
          ds2_transparency=0.65,
          x_locator='day',
          major_day_locator=5)

# 3c) 1 present weather (ww) - synoptic observations - and SWE of high-end GNSS
f.plot_ds(dest_path, '3c1_SWE_GNSS_ref_ww',
          fig_title='SWE of high-end GNSS and present weather (ww)',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-50, 800),
          y2tick_interval=5,
          y_axis2=True,
          y2_label='present weather (ww)',
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6=synop['ww'],
          ds6_label='present weather',
          ds6_color='blue',
          ds6_transparency=0.65,
          ds6_yaxis=2)
# 3c) 2 present weather: snowfall and SWE of high-end GNSS
f.plot_ds(dest_path, '3c2_SWE_GNSS_ref_ww_snow',
          fig_title='SWE of high-end GNSS and present weather',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-50, 800),
          y2_lim=(69, 81),
          legend2_position=(0.68, 0.88),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6_bar=synop['ww'],
          ds6_label='present weather',
          ds6_color='blue',
          ds6_transparency=0.85,
          ds6_yaxis=2)
# 3c) 3a present weather: snowdrift and SWE of high-end GNSS
f.plot_ds(dest_path, '3c3a_SWE_GNSS_ref_ww_snow',
          fig_title='SWE of high-end GNSS and present weather',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-50, 800),
          y2_lim=(35, 41),
          legend2_position=(0.68, 0.88),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6_bar=synop.ww[synop.ww<40],
          ds6_label='present weather',
          ds6_color='blue',
          ds6_transparency=0.9,
          ds6_yaxis=2)
# 3c) 3b CUT: Snowdrift and high-end GNSS refractometry SWE to beginning period
f.plot_ds(dest_path, '3c3b_SWE_GNSS_ref_ww_snow_CUT',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(9, 8),
          x_lim=(dt.date(2021, 12, 7), dt.date(2022, 3, 7)),
          y_lim=(-10, 200),
          ytick_interval=20,
          y2_lim=(35, 41),
          legend_position=(0.01, 0.98),
          legend2_position=(0.68, 0.98),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          x_locator='day',
          major_day_locator=7,
          x_stepsize=7,
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6_bar=synop.ww[synop.ww<40],
          ds6_label='weather',
          ds6_color='blue',
          ds6_transparency=0.9,
          ds6_yaxis=2)


# 3d) synoptic observations (ww) and GNSS-IR snow height (CUT to beginning period)
# 3d) 1 snowdrift
f.plot_ds(dest_path, '3d1_sh_GNSSir_ref_ww_snow_CUT',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(9, 8),
          x_lim=(dt.date(2021, 12, 7), dt.date(2022, 3, 7)),
          y_lim=(-10, 80),
          ytick_interval=20,
          y_label='snow height [cm]',
          y2_lim=(35, 41),
          legend_position=(0.01, 0.98),
          legend2_position=(0.68, 0.98),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          x_locator='day',
          major_day_locator=7,
          x_stepsize=7,
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10,
          ds9_yaxis=1,
          ds6=laser_daily.dsh/10,
          ds7_bar=synop.ww[synop.ww<40],
          ds7_label='weather',
          ds7_color='blue',
          ds7_transparency=0.9,
          ds7_yaxis=2)
# 3d) 2 snowfall
f.plot_ds(dest_path, '3d2_sh_GNSSir_ref_ww_snow_CUT',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          fig_size=(9, 8),
          x_lim=(dt.date(2021, 12, 7), dt.date(2022, 3, 7)),
          y_lim=(-10, 80),
          ytick_interval=20,
          y_label='snow height [cm]',
          y2_lim=(69, 81),
          legend_position=(0.01, 0.98),
          legend2_position=(0.68, 0.98),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          x_locator='day',
          major_day_locator=7,
          x_stepsize=7,
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10,
          ds9_yaxis=1,
          ds6=laser_daily.dsh/10,
          ds7_bar=synop.ww,
          ds7_label='weather',
          ds7_color='blue',
          ds7_transparency=0.9,
          ds7_yaxis=2)

# 3e) synoptic observations (ww) and GNSS-RR density
# 3e) 1 Snowdrift
f.plot_ds(dest_path, '3e1_density_GNSS_ref_ww_snow',
          fig_title='SWE of high-end GNSS and present weather',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          legend2_position=(0.68, 0.88),
          x_lim=(dt.date(2022, 3, 1), dt.date(2023, 8, 31)),
          y2_lim=(35, 41),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          y_lim=(70, 450),
          y_label='density [kg/m³]',
          ytick_interval=50,
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds2=density_LBLR_laser,
          ds2_label='LBLR / laser',
          ds2_color='darkseagreen',
          ds4=density_JBLR_laser,
          ds4_label='JBLR / laser',
          ds4_color='forestgreen',
          ds6_bar=synop.ww[synop.ww<40],
          ds6_label='present weather',
          ds6_color='blue',
          ds6_transparency=0.9,
          ds6_yaxis=2)
# 3e) 2 Snowfall
f.plot_ds(dest_path, '3e2_density_GNSS_ref_ww_snow',
          fig_title='SWE of high-end GNSS and present weather',
          create_plot=plot_3_GNSS_MOB,
          save=save_plots,
          x_lim=(dt.date(2022, 3, 1), dt.date(2023, 8, 31)),
          y2_lim=(69, 81),
          legend2_position=(0.68, 0.88),
          y2tick_interval=1,
          y_axis2=True,
          y2_label='present weather (ww)',
          y_lim=(70, 450),
          y_label='density [kg/m³]',
          ytick_interval=50,
          ds1=density_LBLR.dropna(),
          ds1_label='GNSS-RR (LBLR)',
          ds3=density_JBLR.dropna(),
          ds3_label='GNSS-RR (JBLR)',
          ds2=density_LBLR_laser,
          ds2_label='LBLR / laser',
          ds2_color='darkseagreen',
          ds4=density_JBLR_laser,
          ds4_label='JBLR / laser',
          ds4_color='forestgreen',
          ds6_bar=synop['ww'],
          ds6_label='present weather',
          ds6_color='blue',
          ds6_transparency=0.85,
          ds6_yaxis=2)


''' 15. 4) Snow height of GNSS-IR and reference sensors '''

# 4a) ONLY reference observations (snow pit "manual", laser, buoy, stakefield "poles")
f.plot_ds(dest_path,
          '4a_SHM_only_REF_above_Ant',
          fig_title='Snow Height above Antenna (Reference Sensors)',
          create_plot=plot_4_dsh,
          save=save_plots,
          x_lim=xlim_dates,
          y_label='snow height [cm]',
          y_lim=(-20, 240),
          ytick_interval=20,
          ds6=laser_daily['dsh'].dropna()/10,
          ds6_std=laser_daily['dsh_std'].dropna()/10,
          ds7=buoy_daily['dsh_mean'].dropna()/10,
          ds7_std=buoy_daily['dsh_std'].dropna()/10,
          ds8=poles_daily['sh_mean'].dropna()/10,
          ds8_std=poles_daily['sh_std'].dropna()/10)

# 4b) GNSS-IR and reference observations (laser, buoy, stakefield "poles")
f.plot_ds(dest_path,
          '4b_SHM_above_Ant_GNSSIR_REF',
          fig_title='Snow Height above Antenna',
          create_plot=plot_4_dsh,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-20, 240),
          y_label='snow height [cm]',
          ytick_interval=20,
          ds6=laser_15min['dsh'].dropna()/10,
          ds6_std=laser_15min['dsh_std'].dropna()/10,
          ds7=buoy_daily['dsh_mean'].dropna()/10,
          ds7_std=buoy_daily['dsh_std'].dropna()/10,
          ds8=poles_daily['sh_mean'].dropna()/10,
          ds8_std=poles_daily['sh_std'].dropna()/10,
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10,
          ds9_yaxis=1)

# 4d) Deviation of GNSS-Ir from reference sensors
# 4d) 1 GNSS-IR daviation from laser
f.plot_ds(dest_path,
          '4d1a_deviation_GNSSIR_laser',
          fig_title='a) Snow height observation comparison',
          create_plot=plot_4_dsh_deviation,
          save=save_plots,
          y_lim=(-30, 30),
          ytick_interval=5,
          x_lim=xlim_dates,
          y_label='∆h (GNSS-IR - laser) [cm]',
          ds6=GNSSIR_laser_R.dropna(),
          ds6_label=None)
f.plot_ds(dest_path,
          '4d1b_deviation_GNSSIR_laser_percentual',
          fig_title='b) Snow height observation comparison',
          create_plot=plot_4_dsh_deviation,
          save=save_plots,
          y_lim=(-50, 50),
          ytick_interval=10,
          x_lim=xlim_dates,
          y_label='∆h (GNSS-IR - laser) [%]',
          ds6=GNSSIR_laser_R_p.dropna(),
          ds6_label=None)
# 4d) 2 GNSS-IR daviation from snow buoy
f.plot_ds(dest_path,
          '4d2a_deviation_GNSSIR_buoy',
          fig_title='a) Snow height observation comparison',
          create_plot=plot_4_dsh_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-30, 30),
          ytick_interval=5,
          y_label='∆h (GNSS-IR - snow buoy) [cm]',
          ds7=GNSSIR_buoy_R.dropna(),
          ds7_label=None)
f.plot_ds(dest_path,
          '4d2b_deviation_GNSSIR_buoy_percentual',
          fig_title='b) Snow height observation comparison',
          create_plot=plot_4_dsh_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-50, 50),
          ytick_interval=10,
          y_label='∆h (GNSS-IR - snow buoy) [cm] [%]',
          ds7=GNSSIR_buoy_R_p.dropna(),
          ds7_label=None)
# 4d) 3 GNSS-IR daviation from poles
f.plot_ds(dest_path,
          '4d3a_deviation_GNSSIR_stake',
          fig_title='a) Snow height observation comparison',
          create_plot=plot_4_dsh_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-30, 30),
          ytick_interval=5,
          y_label='∆h (GNSS-IR - stake field) [cm]',
          ds8=GNSSIR_poles_R.dropna(),
          ds8_label=None,
          ds8_linestyle=' ',
          ds8_marker='o')
f.plot_ds(dest_path,
          '4d3b_deviation_GNSSIR_stake_percentual',
          fig_title='b) Snow height observation comparison',
          create_plot=plot_4_dsh_deviation,
          save=save_plots,
          x_lim=xlim_dates,
          y_lim=(-50, 50),
          ytick_interval=10,
          y_label='∆h (GNSS-IR - stake field) [%]',
          ds8=GNSSIR_poles_R_p.dropna(),
          ds8_label=None,
          ds8_linestyle=' ',
          ds8_marker='o')


''' 15. 5) GNSS-refractometry and reflectometry results '''

# 5a) 1 (with space below for phase-insert) SWE of High-End GNSS and Acc of GNSS-IR and laser
f.plot_ds(dest_path, '5a1_SWE_GNSS_SHM_ref',
          fig_title='SWE of high-end GNSS and Snow Height of GNSS-IR',
          create_plot=plot_5_GNSS_RR,
          save=save_plots,
          fig_size=(15, 10),
          y_axis2=True,
          legend2_position=(0.8, 0.4),
          x_lim=xlim_dates,
          y_lim=(-150, 800),
          y2_lim=(-45, 240),
          ytick_interval=50,
          y2tick_interval=15,
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6=laser_15min['dsh'].dropna() / 10,
          ds6_std=laser_15min['dsh_std'].dropna() / 10,
          ds6_yaxis=2,
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10)
# 5a) 2 high-end GNSS refractometry and Acc of GNSS-IR and laser
f.plot_ds(dest_path, '5a2_SWE_GNSS_SHM_ref',
          fig_title='SWE of high-end GNSS and snow height above antenna',
          create_plot=plot_5_GNSS_RR,
          save=save_plots,
          y_lim=(-50, 800),
          y_axis2=True,
          y2_lim=(-15, 240),
          ytick_interval=50,
          y2tick_interval=15,
          legend2_position=(0.8, 0.3),
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6=laser_15min['dsh'].dropna() / 10,
          ds6_std=laser_15min['dsh_std'].dropna() / 10,
          ds6_yaxis=2,
          ds9=gnssir_acc_daily/10,
          ds9_std=gnssir_acc_daily_std/10)
# 5a) 3 Plot SWE of low-cost GNSS and Acc of GNSS-IR
f.plot_ds(dest_path, '5a3_SWE_GNSS_SHM_ref',
          fig_title='SWE of low-cost GNSS and Snow Height of GNSS-IR',
          create_plot=plot_5_GNSS_RR,
          save=save_plots,
          y_axis2=True,
          x_lim=xlim_dates,
          y_lim=(-50, 800),
          y2_lim=(-50, 240),
          y2tick_interval=15,
          ytick_interval=50,
          legend2_position=(0.8, 0.3),
          ds2=swe_gnss_daily_LBUR.dropna(),
          ds2_std=std_gnss_daily_LBUR.dropna(),
          ds4=swe_gnss_daily_JBUR.dropna(),
          ds4_std=std_gnss_daily_JBUR.dropna(),
          ds6=laser_15min['dsh'].dropna() / 10,
          ds6_std=laser_15min['dsh_std'].dropna() / 10,
          ds6_yaxis=2,
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10)
# 5b) 1 Plot SWE of High-End GNSS and Acc of GNSS-IR and reference sensors
f.plot_ds(dest_path, '5b_SWE_GNSS_SHM_ref',
          fig_title='SWE of GNSS and Snow Height of GNSS-IR and Reference Sensors',
          create_plot=plot_5_GNSS_RR,
          save=save_plots,
          y_axis2=True,
          y_lim=(-50, 800),
          y2_lim=(-15, 240),
          y2tick_interval=15,
          ytick_interval=50,
          legend2_position=(0.8, 0.45),
          ds1=swe_gnss_daily_LBLR.dropna(),
          ds1_std=std_gnss_daily_LBLR.dropna(),
          ds3=swe_gnss_daily_JBLR.dropna(),
          ds3_std=std_gnss_daily_JBLR.dropna(),
          ds6=laser_15min['dsh'].dropna() / 10,
          ds6_std=laser_15min['dsh_std'].dropna() / 10,
          ds6_yaxis=2,
          ds7=buoy_daily['dsh_mean'].dropna() / 10,
          ds7_std=buoy_daily['dsh_std'].dropna() / 10,
          ds7_yaxis=2,
          ds8=poles_daily['sh_mean'].dropna() / 10,
          ds8_std=poles_daily['sh_std'].dropna() / 10,
          ds8_yaxis=2,
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10)
# 5b) 2 CUT to beginning period (onset of observation and installation date)
f.plot_ds(dest_path, '5b_SWE_GNSS_SHM_GNSSIR_ref',
          fig_title=' ',
          create_plot=plot_5_GNSS_RR,
          save=save_plots,
          x_lim=(dt.date(2021, 11, 12), dt.date(2021, 12, 31)),
          x_locator='day',
          major_day_locator=5,
          x_stepsize=5,
          legend_position=(0.02, 0.97),
          legend_title='mass:',
          legend2_position=(0.8, 0.97),
          legend2_title='snow height:',
          y_axis2=True,
          y_lim=(0, 300),
          y2_lim=(-10, 80),
          y2tick_interval=20,
          ytick_interval=50,
          plot_date_lines=True,
          ds1=swe_gnss_LBLR.dropna(),
          ds1_std=std_gnss_LBLR.dropna(),
          ds3=swe_gnss_JBLR.dropna(),
          ds3_std=std_gnss_JBLR.dropna(),
          ds6=laser_15min['dsh'].dropna() / 10,
          ds6_std=laser_15min['dsh_std'].dropna() / 10,
          ds6_yaxis=2,
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10)


# 5c) 1 plot snow mass (SWE) versus snow depth
f.plot_ds(dest_path, '5c1_SWE_GNSS_over_SH_GNSSIR',
          fig_title='SWE versus Snow Height',
          create_plot=plot_5_refracto_over_reflecto,
          save=save_plots,
          fig_size=(8, 10.9),
          y_label='mass [kg/m²]',
          x_label='snow height [m]',
          x_lim=(0, 1.4),
          y_lim=(0, 450),
          xtick_interval=0.2,
          ytick_interval=50,
          x_datetime=False,
          ds1=swe_over_sh['swe_gnss_daily_LBLR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_fit=exp_fit_LBLR,
          ds1_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_LBLR[2]),
          ds2=swe_over_sh['swe_gnss_daily_LBUR'],
          ds2_marker='o',
          ds2_linestyle=' ',
          ds2_fit=exp_fit_LBUR,
          ds2_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_LBUR[2]),
          ds3=swe_over_sh['swe_gnss_daily_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=exp_fit_JBLR,
          ds3_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_JBLR[2]),
          ds4=swe_over_sh['swe_gnss_daily_JBUR'],
          ds4_marker='o',
          ds4_linestyle=' ',
          ds4_fit=exp_fit_JBUR,
          ds4_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_JBUR[2]))
# 5c) 2 plot high-end GNSS snow mass (SWE) versus snow height
f.plot_ds(dest_path, '5c2_SWE_heGNSS_over_SH_GNSSIR',
          fig_title='SWE versus Snow Height',
          create_plot=plot_5_refracto_over_reflecto,
          save=save_plots,
          fig_size=(8, 10.9),
          y_label='mass [kg/m²]',
          x_label='snow height [m]',
          x_lim=(0, 1.4),
          y_lim=(0, 450),
          xtick_interval=0.2,
          ytick_interval=50,
          x_datetime=False,
          ds1=swe_over_sh['swe_gnss_daily_LBLR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_fit=exp_fit_LBLR,
          ds1_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_LBLR[2]),
          ds3=swe_over_sh['swe_gnss_daily_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=exp_fit_JBLR,
          ds3_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_JBLR[2]))
# 5c) 3 plot low-cost GNSS snow mass (SWE) versus snow height
f.plot_ds(dest_path, '5c3_SWE_lcGNSS_over_SH_GNSSIR',
          fig_title='SWE versus Snow Height',
          create_plot=plot_5_refracto_over_reflecto,
          save=save_plots,
          fig_size=(8, 10.9),
          y_label='mass [kg/m²]',
          x_label='snow height [m]',
          x_lim=(0, 1.4),
          y_lim=(0, 450),
          xtick_interval=0.2,
          ytick_interval=50,
          x_datetime=False,
          ds2=swe_over_sh['swe_gnss_daily_LBUR'],
          ds2_marker='o',
          ds2_linestyle=' ',
          ds2_fit=exp_fit_LBUR,
          ds2_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_LBUR[2]),
          ds4=swe_over_sh['swe_gnss_daily_JBUR'],
          ds4_marker='o',
          ds4_linestyle=' ',
          ds4_fit=exp_fit_JBUR,
          ds4_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_JBUR[2]))
# 5c) 2b plot high-end GNSS snow mass (SWE) versus snow height from 15.02.22 onwards
f.plot_ds(dest_path, '5c2_SWE_heGNSS_over_SH_GNSSIR_0103',
          fig_title='SWE versus Snow Height',
          create_plot=plot_5_refracto_over_reflecto,
          save=save_plots,
          fig_size=(8, 10.9),
          y_label='mass [kg/m²]',
          x_label='snow height [m]',
          x_lim=(0, 1.4),
          y_lim=(0, 450),
          xtick_interval=0.2,
          ytick_interval=50,
          x_datetime=False,
          ds1=swe_over_sh_0103['swe_gnss_daily_LBLR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_fit=exp_fit_LBLR_0103,
          ds1_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_LBLR_0103[2]),
          ds3=swe_over_sh_0103['swe_gnss_daily_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=exp_fit_JBLR_0103,
          ds3_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_JBLR_0103[2]))
# 5c) 3b plot low-cost GNSS snow mass (SWE) versus snow height from 15.02.22 onwards
f.plot_ds(dest_path, '5c3_SWE_lcGNSS_over_SH_GNSSIR_0103',
          fig_title='SWE versus Snow Height',
          create_plot=plot_5_refracto_over_reflecto,
          save=save_plots,
          fig_size=(8, 10.9),
          y_label='mass [kg/m²]',
          x_label='snow height [m]',
          x_lim=(0, 1.4),
          y_lim=(0, 450),
          xtick_interval=0.2,
          ytick_interval=50,
          x_datetime=False,
          ds2=swe_over_sh_0103['swe_gnss_daily_LBUR'],
          ds2_marker='o',
          ds2_linestyle=' ',
          ds2_fit=exp_fit_LBUR_0103,
          ds2_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_LBUR_0103[2]),
          ds4=swe_over_sh_0103['swe_gnss_daily_JBUR'],
          ds4_marker='o',
          ds4_linestyle=' ',
          ds4_fit=exp_fit_JBUR_0103,
          ds4_fit_label="fit: {:.0f}  exp({:.2f}x){:.0f}".format(*exp_fit_JBUR_0103[2]))


''' 15. 6) Density of GNSS-RR and reference '''

# 6a) Plot surface snow density of ALL
f.plot_ds(dest_path, '6a1_density_highend',
          fig_title='Density of Snowpack above Antenna',
          create_plot=plot_6_density,
          save=save_plots,
          legend_position=(0.75, 0.4),
          x_lim=(dt.date(2021, 12, 1), dt.date(2023, 8, 15)),
          y_lim=(0, 650),
          ytick_interval=50,
          y_label='density [kg/m³]',
          ds1=density_LBLR.dropna(),
          ds1_label='GNSS-RR (LBLR)',
          ds3=density_JBLR.dropna(),
          ds3_label='GNSS-RR (JBLR)',
          ds2=density_LBLR_laser,
          ds2_label='LBLR / laser',
          ds2_color='darkseagreen',
          ds4=density_JBLR_laser,
          ds4_label='JBLR / laser',
          ds4_color='forestgreen',
          ds5=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna(),
          ds5_err=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna()*0.15)
# cut out
f.plot_ds(dest_path, '6a1b_density_highend_cut',
          create_plot=plot_6_density,
          save=save_plots,
          fig_size=(15, 7),
          y_axis2=True,
          legend_position=(0.01, 0.98),
          legend2_position=(0.8, 0.4),
          x_lim=(dt.date(2022, 5, 1), dt.date(2022, 10, 30)),
          y_lim=(100, 350),
          y2_lim=(40, 140),
          ytick_interval=50,
          y2tick_interval=20,
          y_label='density [kg/m³]',
          ds1=density_LBLR.dropna(),
          ds1_label='GNSS-RR (LBLR)',
          ds3=density_JBLR.dropna(),
          ds3_label='GNSS-RR (JBLR)',
          ds9=gnssir_acc_daily/10,
          ds9_std=gnssir_acc_daily_std/10)

f.plot_ds(dest_path, '6a2_density_lowcost',
          fig_title='Density of Snowpack above Antenna',
          create_plot=plot_6_density,
          save=save_plots,
          legend_position=(0.75, 0.3),
          x_lim=(dt.date(2021, 12, 1), dt.date(2023, 8, 15)),
          y_lim=(0, 650),
          ytick_interval=50,
          y_label='density [kg/m³]',
          ds2=density_LBUR.dropna(),
          ds2_label='GNSS-RR (LBUR)',
          ds4=density_JBUR.dropna(),
          ds4_label='GNSS-RR (JBUR)',
          ds1=density_LBUR_laser,
          ds1_label='LBUR / laser',
          ds1_color='darkseagreen',
          ds3=density_JBUR_laser,
          ds3_label='JBUR / laser',
          ds3_color='forestgreen',
          ds5=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna(),
          ds5_err=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna()*0.15)

# 6b) 1 Plot surface snow density of High-end GNSS and density measurements of snow pit of snow height above antenna
f.plot_ds(dest_path, '6b1_density_above_Ant_HighEnd_snowpit',
          fig_title='Density of Snowpack above Antenna',
          create_plot=plot_6_density,
          save=save_plots,
          legend_position=(0.35, 0.25),
          y_label='density [kg/m³]',
          ytick_interval=50,
          y_lim=(0, 650),
          y_axis2=True,
          y2_lim=(0, 160),
          y2tick_interval=20,
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10,
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds5=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna(),
          ds5_err=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna()*0.15)
# 6b) 2 Plot surface snow density above antenna of High-end GNSS and 1m density of snow pit
f.plot_ds(dest_path, '6b2_density_above_Ant_HighEnd_1m_density_snowpit',
          fig_title='GNSS-RR Density above Antenna and ~1m Snow Pit Density',
          create_plot=plot_6_density,
          save=save_plots,
          y_label='density [kg/m³]',
          ytick_interval=50,
          y_lim=(0, 650),
          y_axis2=True,
          y2_lim=(0, 160),
          y2tick_interval=20,
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10,
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds5=manual_NEW['density / snow pit profile [kg/m³]'].dropna(),
          ds5_label='density (over snow pit depth)')
# 6c) Plot surface snow density of High-end with GNSSIR and cutting date
f.plot_ds(dest_path, '6c_density_HighEnd_show_cutting_date',
          fig_title='GNSS-RR Density of Snowpack above Antenna',
          create_plot=plot_6_density,
          save=save_plots,
          y_label='density [kg/m³]',
          ytick_interval=50,
          y_lim=(0, 600),
          y_axis2=True,
          y2_lim=(0, 160),
          y2tick_interval=20,
          ax2_hline_value=gnssir_acc_daily['2022-02-19'] / 10,
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10,
          plot_vline=dt.datetime(2022, 2, 19))

# 6d) 1 cut out: surface snow density of high-end GNSS-RR and Acc of GNSS-IR
f.plot_ds(dest_path, '6d1_density_HighEnd_Acc_GNSSIR_cut',
          fig_title='GNSS-RR density of snowpack above antenna and GNSS-IR snow height',
          create_plot=plot_6_density,
          save=save_plots,
          y_label='density [kg/m³]',
          ytick_interval=50,
          y_axis2=True,
          y2tick_interval=20,
          y_lim=(50, 400),
          y2_lim=(20, 150),
          x_lim=(dt.date(2022, 2, 19), dt.date(2023, 4, 15)),
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10)
# 6d) 1b cut out: surface snow density of high-end GNSS-RR and Acc of GNSS-IR + snow pit
f.plot_ds(dest_path, '6d1_density_HighEnd_Acc_GNSSIR_cut',
          fig_title='Mean Density of Snowpack above Antenna',
          create_plot=plot_6_density,
          save=save_plots,
          fig_size=(15, 10),
          legend_position=(0.22, 0.35),
          legend2_position=(0.8, 0.35),
          y_label='density [kg/m³]',
          ytick_interval=50,
          y_axis2=True,
          y2tick_interval=10,
          y_lim=(-50, 650),
          y2_lim=(10, 150),
          x_lim=(dt.date(2022, 2, 19), dt.date(2023, 4, 15)),
          ds1=density_LBLR.dropna(),
          ds3=density_JBLR.dropna(),
          ds9=gnssir_acc_daily.dropna()/10,
          ds9_std=gnssir_acc_daily_std.dropna()/10,
          ds5=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna(),
          ds5_err=manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].dropna()*0.15)
# 6d) 2 cut out: surface snow density of high-end GNSS-RR and Acc of GNSS-IR and laser
f.plot_ds(dest_path, '6d2b_density_HighEnd_Acc_GNSSIR_laser_cut',
          create_plot=plot_6_density,
          save=save_plots,
          fig_size=(15, 10),
          legend_position=(0.01, 0.95),
          legend2_position=(0.73, 0.53),
          legend3_position=(0.73, 0.33),
          legend_title='mass:',
          legend2_title='snow height:',
          legend3_title='density:',
          y_axis2=True,
          y_axis3=True,
          ytick_interval=50,
          y2tick_interval=15,
          y3tick_interval=25,
          y_lim=(-100, 700),
          y2_lim=(-15, 225),
          y3_lim=(0, 400),
          x_lim=(dt.date(2022, 2, 19), dt.date(2023, 8, 31)),
          ds1=swe_gnss_daily_LBLR,
          ds1_std=std_gnss_daily_LBLR,
          ds1_label='GNSS refractometry (LBLR)',
          ds3=density_LBLR.dropna(),
          ds3_yaxis=3,
          ds3_color='crimson',
          ds3_linestyle='--',
          ds3_label='GNSS-RR (LBLR)',
          ds4=swe_gnss_daily_LBLR / laser_daily.dsh * 1000,
          ds4_yaxis=3,
          ds4_label='LBLR / laser',
          ds4_color='darkseagreen',
          ds4_transparency=0.1,
          ds9=gnssir_acc_daily.dropna() / 10,
          ds9_std=gnssir_acc_daily_std.dropna() / 10,
          ds6=laser_daily['dsh'].dropna() / 10,
          ds6_yaxis=2)

# 6e) 1 deviation between GNSS-RR and snow pit densities
f.plot_ds(dest_path,
          '6e1a_deviation_GNSSRR_manual',
          fig_title='a) Density estimate comparison (absolute deviation)',
          create_plot=plot_6_density_deviation,
          save=save_plots,
          legend_position=(0.6, 0.3),
          x_lim=xlim_dates,
          y_lim=(-450, -0),
          ytick_interval=50,
          y_label='∆density (snow pit - GNSS-RR) [kg/m³]',
          ds1=GNSSRR_LBLR_manual_R.dropna(),
          ds1_linestyle=' ',
          ds1_marker='+',
          ds1_markersize=10,
          ds2=GNSSRR_LBUR_manual_R.dropna(),
          ds2_linestyle=' ',
          ds2_marker='+',
          ds2_markersize=10,
          ds3=GNSSRR_JBLR_manual_R.dropna(),
          ds3_linestyle=' ',
          ds3_marker='+',
          ds3_markersize=10,
          ds4=GNSSRR_JBUR_manual_R.dropna(),
          ds4_linestyle=' ',
          ds4_marker='+',
          ds4_markersize=10)
f.plot_ds(dest_path,
          '6e1b_deviation_GNSSRR_manual_percentual',
          fig_title='b) Density estimate comparison (relative deviation)',
          create_plot=plot_6_density_deviation,
          save=save_plots,
          legend_position=(0.6, 0.3),
          x_lim=xlim_dates,
          y_lim=(-100, 0),
          ytick_interval=10,
          y_label='∆density (snow pit - GNSS-RR) [%]',
          ds1=GNSSRR_LBLR_manual_R_p.dropna(),
          ds1_linestyle=' ',
          ds1_marker='+',
          ds1_markersize=10,
          ds2=GNSSRR_LBUR_manual_R_p.dropna(),
          ds2_linestyle=' ',
          ds2_marker='+',
          ds2_markersize=10,
          ds3=GNSSRR_JBLR_manual_R_p.dropna(),
          ds3_linestyle=' ',
          ds3_marker='+',
          ds3_markersize=10,
          ds4=GNSSRR_JBUR_manual_R_p.dropna(),
          ds4_linestyle=' ',
          ds4_marker='+',
          ds4_markersize=10)
# 6e) 3 deviation of GNSS refractometry / laser from snow pit densities
f.plot_ds(dest_path,
          '6e3a_deviation_GNSSrefr_laser_manual',
          fig_title='a) Density estimate comparison (absolute deviation)',
          create_plot=plot_6_density_deviation,
          save=save_plots,
          legend_position=(0.6, 0.3),
          x_lim=xlim_dates,
          y_lim=(-450, -0),
          ytick_interval=50,
          y_label='∆density (snow pit - GNSS refractometry/laser) [kg/m³]',
          ds1=GNSSRR_LBLR_laser_manual_R.dropna(),
          ds1_linestyle=' ',
          ds1_marker='+',
          ds1_markersize=10,
          ds2=GNSSRR_LBUR_laser_manual_R.dropna(),
          ds2_linestyle=' ',
          ds2_marker='+',
          ds2_markersize=10,
          ds3=GNSSRR_JBLR_laser_manual_R.dropna(),
          ds3_linestyle=' ',
          ds3_marker='+',
          ds3_markersize=10,
          ds4=GNSSRR_JBUR_laser_manual_R.dropna(),
          ds4_linestyle=' ',
          ds4_marker='+',
          ds4_markersize=10)
f.plot_ds(dest_path,
          '6e3b_deviation_GNSSrefr_laser_manual_percentual',
          fig_title='b) Density estimate comparison (relative deviation)',
          create_plot=plot_6_density_deviation,
          save=save_plots,
          legend_position=(0.6, 0.3),
          x_lim=xlim_dates,
          y_lim=(-100, 0),
          ytick_interval=10,
          y_label='∆density (snow pit - GNSS refractometry/laser) [%]',
          ds1=GNSSRR_LBLR_laser_manual_R_p.dropna(),
          ds1_linestyle=' ',
          ds1_marker='+',
          ds1_markersize=10,
          ds2=GNSSRR_LBUR_laser_manual_R_p.dropna(),
          ds2_linestyle=' ',
          ds2_marker='+',
          ds2_markersize=10,
          ds3=GNSSRR_JBLR_laser_manual_R_p.dropna(),
          ds3_linestyle=' ',
          ds3_marker='+',
          ds3_markersize=10,
          ds4=GNSSRR_JBUR_laser_manual_R_p.dropna(),
          ds4_linestyle=' ',
          ds4_marker='+',
          ds4_markersize=10)

# 6g) Density error
# 6g) 1 relative error for σ_h=0.1m and different σ_m
f.plot_ds(dest_path,
          '6g1_GNSSrr_density_error_relative',
          create_plot=plot_6_density_error,
          save=save_plots,
          fig_size=(7, 7),
          legend_position=(0.2, 0.98),
          x_datetime=False,
          y_lim=(0, 300),
          ytick_interval=30,
          x_lim=(0, 5),
          xtick_interval=0.5,
          y_label='relative density error [%]',
          x_label='snow height [m]',
          ds1=density_err_h01.rel_m10_h01 * 100,
          ds1_label='$σ_m$ = 10kg/m²',
          ds1_color='tomato',
          ds2=density_err_h01.rel_m30_h01*100,
          ds2_label='$σ_{m}$ = 30kg/m²',
          ds2_color='red',
          ds3=density_err_h01.rel_m50_h01 * 100,
          ds3_label='$σ_{m}$ = 50kg/m²',
          ds3_color='darkred',
          ds4=density_err_h01.rel_m90_h01 * 100,
          ds4_label='$σ_{m}$ = 90kg/m²',
          ds4_color='black')
# 6g) 2 absolute error for σ_h=0.1m and different σ_m
f.plot_ds(dest_path,
          '6g2_GNSSrr_density_error_absolute',
          create_plot=plot_6_density_error,
          save=save_plots,
          fig_size=(7, 7),
          x_datetime=False,
          legend_position=(0.2, 0.98),
          y_lim=(0, 300),
          ytick_interval=30,
          x_lim=(0, 5),
          xtick_interval=0.5,
          y_label='$σ_ρ$ absolute density error [kg/m³]',
          x_label='snow height [m]',
          ds1=density_err_h01.abs_m10_h01,
          ds1_label='$σ_{m}$ = 10kg/m²',
          ds1_color='tomato',
          ds2=density_err_h01.abs_m30_h01,
          ds2_label='$σ_{m}$ = 30kg/m²',
          ds2_color='red',
          ds3=density_err_h01.abs_m50_h01,
          ds3_label='$σ_{m}$ = 50kg/m²',
          ds3_color='darkred',
          ds4=density_err_h01.abs_m90_h01,
          ds4_label='$σ_{m}$ = 90kg/m²',
          ds4_color='black')
# 6g) 3 relative error for different σ_h and σ_m=30kg/m²
f.plot_ds(dest_path,
          '6g3_GNSSrr_density_error_relative',
          create_plot=plot_6_density_error,
          save=save_plots,
          fig_size=(7, 7),
          legend_position=(0.2, 0.98),
          x_datetime=False,
          y_lim=(0, 300),
          ytick_interval=30,
          x_lim=(0, 5),
          xtick_interval=0.5,
          y_label='relative density error [%]',
          x_label='snow height [m]',
          ds1=density_err_m30.rel_m30_h01 * 100,
          ds1_label='$σ_h$ = 0.1 m',
          ds1_color='tomato',
          ds2=density_err_m30.rel_m30_h02*100,
          ds2_label='$σ_{h}$ = 0.2 m',
          ds2_color='red',
          ds3=density_err_m30.rel_m30_h03 * 100,
          ds3_label='$σ_{h}$ = 0.3 m',
          ds3_color='darkred',
          ds4=density_err_m30.rel_m30_h04 * 100,
          ds4_label='$σ_{h}$ = 0.4 m',
          ds4_color='black')
# 6g) 4 absolute error for different σ_h and σ_m=30kg/m²
f.plot_ds(dest_path,
          '6g4_GNSSrr_density_error_absolute',
          create_plot=plot_6_density_error,
          save=save_plots,
          fig_size=(7, 7),
          x_datetime=False,
          legend_position=(0.2, 0.98),
          y_lim=(0, 300),
          ytick_interval=30,
          x_lim=(0, 5),
          xtick_interval=0.5,
          y_label='$σ_ρ$ absolute density error [kg/m³]',
          x_label='snow height [m]',
          ds1=density_err_m30.abs_m30_h01,
          ds1_label='$σ_h$ = 0.1 m',
          ds1_color='tomato',
          ds2=density_err_m30.abs_m30_h02,
          ds2_label='$σ_{h}$ = 0.2 m',
          ds2_color='red',
          ds3=density_err_m30.abs_m30_h03,
          ds3_label='$σ_{h}$ = 0.3 m',
          ds3_color='darkred',
          ds4=density_err_m30.abs_m30_h04,
          ds4_label='$σ_{h}$ = 0.4 m',
          ds4_color='black')
# 6g) 5 absolute error for different fixed densities
f.plot_ds(dest_path,
          '6g5_GNSSrr_density_error_absolute',
          create_plot=plot_6_density_error,
          save=save_plots,
          fig_size=(7, 7),
          x_datetime=False,
          legend_position=(0.2, 0.98),
          y_lim=(0, 300),
          ytick_interval=30,
          x_lim=(0, 5),
          xtick_interval=0.5,
          y_label='$σ_ρ$ absolute density error [kg/m³]',
          x_label='snow height [m]',
          ds1=density_err.abs_err_50,
          ds1_label='ρ = 50 kg/m³',
          ds1_color='tomato',
          ds2=density_err.abs_err_150,
          ds2_label='ρ = 150 kg/m³',
          ds2_color='red',
          ds3=density_err.abs_err_300,
          ds3_label='ρ = 300 kg/m³',
          ds3_color='darkred',
          ds4=density_err.abs_err_550,
          ds4_label='ρ = 550 kg/m³',
          ds4_color='black')


''' 15. 7) density versus snow height '''

# 7a) plot snow density versus snow depth - high-end
f.plot_ds(dest_path, '7a_depth_over_density_heGNSS',
          create_plot=plot_7_density_over_dsh,
          save=save_plots,
          fig_size=(8, 11),
          legend_position=(0.02, 0.97),
          x_lim=(0, 1.5),
          y_lim=(50, 400),
          xtick_interval=0.2,
          ytick_interval=50,
          y_label='density [kg/m³]',
          x_label='snow height [m]',
          x_datetime=False,
          ds1=density_over_sh['density_LBLR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_fit=fit_sh_density_LBLR,
          ds1_fit_label="fit: Δm/Δh",
          ds2_fit=fit_derivative_LBLR,
          ds2_fit_label='derivative: 128  exp(1.22x)',
          ds2_fit_linestyle='--',
          ds2_color='crimson',
          ds3=density_over_sh['density_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=fit_sh_density_JBLR,
          ds3_fit_label="fit: Δm/Δh",
          ds4_fit=fit_derivative_JBLR,
          ds4_fit_label='derivative: 125  exp(1.24x)',
          ds4_fit_linestyle='--',
          ds4_color='dodgerblue')
# 7b) plot snow density versus snow depth - low-cost
f.plot_ds(dest_path, '7b_depth_over_density_lcGNSS',
          create_plot=plot_7_density_over_dsh,
          save=save_plots,
          fig_size=(8, 11),
          legend_position=(0.02, 0.97),
          x_lim=(0, 1.5),
          y_lim=(50, 400),
          xtick_interval=0.2,
          ytick_interval=50,
          y_label='density [kg/m³]',
          x_label='snow height [m]',
          x_datetime=False,
          ds1=density_over_sh['density_LBUR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_color='salmon',
          ds1_fit=fit_sh_density_LBUR,
          ds1_fit_label="fit: Δm/Δh",
          ds2_fit=fit_derivative_LBUR,
          ds2_fit_label='derivative: 152  exp(0.84x)',
          ds2_fit_linestyle='--',
          ds2_color='salmon',
          ds3=density_over_sh['density_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=fit_sh_density_JBUR,
          ds3_color='deepskyblue',
          ds3_fit_label="fit: Δm/Δh",
          ds4_fit=fit_derivative_JBUR,
          ds4_fit_label='derivative: 175  exp(0.69x)',
          ds4_fit_linestyle='--',
          ds4_color='deepskyblue')
# starting from 01.03.2022
# 7a) 2 plot snow density versus snow depth - high-end
f.plot_ds(dest_path, '7a2_depth_over_density_heGNSS_0103',
          create_plot=plot_7_density_over_dsh,
          save=save_plots,
          fig_size=(8, 11),
          legend_position=(0.02, 0.97),
          x_lim=(0, 1.5),
          y_lim=(50, 400),
          xtick_interval=0.2,
          ytick_interval=50,
          y_label='density [kg/m³]',
          x_label='snow height [m]',
          x_datetime=False,
          ds1=density_over_sh_0103['density_LBLR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_fit=fit_sh_density_LBLR_0103,
          ds1_fit_label="fit: Δm/Δh",
          ds2_fit=fit_derivative_LBLR_0103,
          ds2_fit_label='derivative: 131  exp(1.20x)',
          ds2_fit_linestyle='--',
          ds2_color='crimson',
          ds3=density_over_sh_0103['density_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=fit_sh_density_JBLR_0103,
          ds3_fit_label="fit: Δm/Δh",
          ds4_fit=fit_derivative_JBLR_0103,
          ds4_fit_label='derivative: 120  exp(1.30x)',
          ds4_fit_linestyle='--',
          ds4_color='dodgerblue')
# 7b) 2 plot snow density versus snow depth - low-cost
f.plot_ds(dest_path, '7b2_depth_over_density_lcGNSS_0103',
          create_plot=plot_7_density_over_dsh,
          save=save_plots,
          fig_size=(8, 11),
          legend_position=(0.02, 0.97),
          x_lim=(0, 1.5),
          y_lim=(50, 400),
          xtick_interval=0.2,
          ytick_interval=50,
          y_label='density [kg/m³]',
          x_label='snow height [m]',
          x_datetime=False,
          ds1=density_over_sh_0103['density_LBUR'],
          ds1_marker='o',
          ds1_linestyle=' ',
          ds1_color='salmon',
          ds1_fit=fit_sh_density_LBUR_0103,
          ds1_fit_label="fit: Δm/Δh",
          ds2_fit=fit_derivative_LBUR_0103,
          ds2_fit_label='derivative: 149  exp(0.90x)',
          ds2_fit_linestyle='--',
          ds2_color='salmon',
          ds3=density_over_sh_0103['density_JBLR'],
          ds3_marker='o',
          ds3_linestyle=' ',
          ds3_fit=fit_sh_density_JBUR_0103,
          ds3_color='deepskyblue',
          ds3_fit_label="fit: Δm/Δh",
          ds4_fit=fit_derivative_JBUR_0103,
          ds4_fit_label='derivative: 162  exp(0.82x)',
          ds4_fit_linestyle='--',
          ds4_color='deepskyblue')


''' 15. 8) GNSS refractometry footprint '''

# 8) GNSS refractometry radius around rover antenna (footprint)
f.plot_ds(dest_path,
          '8_GNSS_ref_radius',
          create_plot=plot_8_footprint,
          save=save_plots,
          fig_size=(9.3, 8.3),
          x_datetime=False,
          switch_xy=True,
          legend_position=(0.51, 0.57),
          x_lim=(0, 240),
          xtick_interval=20,
          y_lim=(0, 200),
          ytick_interval=20,
          x_label='radius from rover antenna [cm]',
          y_label='snow height [cm]',
          ds1=GNSS_refr_radius_5.r_ds,
          ds1_label='angle of incidence 5°: \n dry snow (n=1.32)',
          ds1_color='lightgreen',
          ds2=GNSS_refr_radius_5.r_ms,
          ds2_label='moist snow (n=1.48)',
          ds2_color='darkseagreen',
          ds3=GNSS_refr_radius_5.r_ws,
          ds3_label='wet snow (n=1.81)',
          ds3_color='forestgreen',
          ds4=GNSS_refr_radius_5.r_vws,
          ds4_label='very wet snow (n=2.3)\n',
          ds4_color='darkgreen',
          ds5=GNSS_refr_radius_90.r_ds,
          ds5_label='angle of incidence 90°: \n dry snow (n=1.32)',
          ds5_color='lightskyblue',
          ds5_linestyle='-',
          ds5_marker='',
          ds6=GNSS_refr_radius_90.r_ms,
          ds6_label='moist snow (n=1.48)',
          ds6_color='steelblue',
          ds6_linestyle='-',
          ds6_marker='',
          ds7=GNSS_refr_radius_90.r_ws,
          ds7_label='wet snow (n=1.81)',
          ds7_color='blue',
          ds7_linestyle='-',
          ds7_marker='',
          ds8=GNSS_refr_radius_90.r_vws,
          ds8_label='very wet snow (n=2.3)',
          ds8_color='black',
          ds8_linestyle='-',
          ds8_marker='')


''' 15. 9) New snow layer properties'''
for i in new_snow:
    interval = i[0]
    min_acc = i[1]
    new_snow_datetimeindex_LBLR, new_snow_heightindex_LBLR = f.calc_new_snow_density(gnssir_acc_daily / 1000, swe_gnss_daily_LBLR.interpolate(), interval=interval, min_acc=min_acc)
    new_snow_datetimeindex_JBLR, new_snow_heightindex_JBLR = f.calc_new_snow_density(gnssir_acc_daily / 1000, swe_gnss_daily_JBLR.interpolate(), interval=interval, min_acc=min_acc)
    # 9a) SWE
    f.plot_ds(dest_path, '9a_' + str(interval) + 'd_' + str(int(min_acc*100)) + 'cm_new_snow_swe',
              fig_title='SWE and Snow Height of New Snow Layer \n with a minimum thickness of ' + str(int(min_acc*100)) + ' cm, accumulated over ' + str(interval) + ' days',
              create_plot=plot_9_new_snow_layer,
              save=save_plots,
              x_lim=(dt.date(2021, 12, 2), dt.date(2023, 4, 25)),
              legend_title='mass:',
              legend2_title='snow height:',
              legend_position=(0.01, 0.85),
              legend2_position=(0.8, 0.85),
              y_label='Δmass [kg/m²/' + str(interval) + 'd]',
              y2_label='Δsnow height [cm/' + str(interval) + 'd]',
              ytick_interval=50,
              y_lim=(0, 400),
              y_axis2=True,
              y2_lim=(0, 80),
              y2tick_interval=10,
              ds1=new_snow_datetimeindex_LBLR.swe,
              ds1_linestyle='',
              ds1_marker='o',
              ds3=new_snow_datetimeindex_JBLR.swe,
              ds3_linestyle='',
              ds3_marker='o',
              ds9_bar=new_snow_datetimeindex_LBLR.h*100,
              ds9_transparency=0.6)
    # 9b) density
    f.plot_ds(dest_path, '9b_' + str(interval) + 'd_' + str(int(min_acc*100)) + 'cm_new_snow_density',
              fig_title='Density and Snow Height of New Snow Layer \n with a minimum thickness of ' + str(int(min_acc*100)) + ' cm, accumulated over ' + str(interval) + ' days',
              create_plot=plot_9_new_snow_layer,
              save=save_plots,
              x_lim=(dt.date(2021, 12, 2), dt.date(2023, 4, 25)),
              legend_title='density:',
              legend2_title='snow height:',
              legend_position=(0.01, 0.85),
              legend2_position=(0.8, 0.85),
              y2_label='Δsnow height [cm/' + str(interval) + 'd]',
              y_label='density of new snow layer [kg/m³]',
              y_lim=(0, 800),
              ytick_interval=100,
              y_axis2=True,
              y2_lim=(0, 80),
              y2tick_interval=10,
              ds1=new_snow_datetimeindex_LBLR.density,
              ds1_linestyle='',
              ds1_marker='+',
              ds1_markersize=10,
              ds3=new_snow_datetimeindex_JBLR.density,
              ds3_linestyle='',
              ds3_marker='+',
              ds3_markersize=10,
              ds9_bar=new_snow_datetimeindex_LBLR.h*100,
              ds9_transparency=0.6)




''' 16. Back up data '''

if solplot_backup is True:
    # copy solutions and plots directories back to server
    f.copy_solplotsdirs(dest_path, backup_path + 'Run_RTKLIB/processing_directory/')

if total_backup is True:
    # copy entire processing directory back to server
    f.copy4backup(dest_path + '../', backup_path)

