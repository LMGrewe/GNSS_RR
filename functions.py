""" 
    Functions to execute: SWE_postproc_RTKLIB_Neumayer.py;
    including necessary preprocessing steps to obtain daily rnx files for processing with RTKLib

    created by: L. Steiner (ORCID: 0000-0002-4958-0849) - original codebase: https://github.com/lasteine/GNSS_RR.git
    revised and expanded by: L.M. Grewe (ORCID: 0009-0009-6533-3432)
    created on: 17.05.2022
    last updated on: 27.10.2023

    requirements:   - install gnssrefl on Linux/Mac (gnssrefl is not working on Windows, see gnssrefl docs)
                    - gnssrefl (https://github.com/kristinemlarson/gnssrefl)
                    - RTKLib (v2.4.3 b34, https://www.rtklib.com/)
                    - gfzrnx (https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:1577894)
                    - jps2rin (http://www.javadgnss.com.cn/products/software/jps2rin.html)
                    - wget
                    - 7zip
                    - path to all programs added to the system environment variables
"""

import subprocess
import os
import glob
import datetime
import shutil
import lzma
import tarfile
import gnsscal
import datetime as dt
import pandas as pd
import numpy as np
import jdcal
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, DayLocator
from matplotlib.ticker import NullFormatter
import matplotlib.pylab as pylab
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.optimize import curve_fit
from termcolor import colored
import requests
import zipfile
import io
from datetime import date
import re
import py7zr
from itertools import chain
import math



""" Define general functions """


def create_folder(dest_path):
    """ create a directory if it is not already existing
    :param dest_path: path and name of new directory
    """
    # Q: create 'temp' directory if not existing
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
        print(colored("\ntemp dir created: %s" % dest_path, 'yellow'))
    else:
        print(colored("\ntemp dir already existing: %s" % dest_path, 'blue'))


def remove_folder(dest_path):
    """ delete temporary directory
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    """
    shutil.rmtree(dest_path)
    print(colored("\n!!! temporary directory removed: %s" % dest_path, 'yellow'))


def get_mjd_int(syyyy, smm, sdd, eyyyy, emm, edd):
    """ calculate start/end mjd using a start and end date of format '2022', '10', '01')
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    """
    start_mjd = jdcal.gcal2jd(str(syyyy), str(smm), str(sdd))[1]
    end_mjd = jdcal.gcal2jd(str(eyyyy), str(emm), str(edd))[1]

    return start_mjd, end_mjd


""" Define preprocessing functions """


def move_files2parentdir(dest_path, f):
    """ move files from temporary preprocessing to main processing directory (parent directory)
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    :param f: file in folder
    """
    # get parent directory
    parent_dir = os.path.dirname(os.path.dirname(dest_path))
    # destination file in parent directory
    dest_path_file = os.path.join(parent_dir, f.split("\\")[-1])
    # move file if it does not already exist
    if not os.path.exists(dest_path_file):
        shutil.move(f, parent_dir)
        print("obs file (%s) moved to parent dir: %s" % (f, dest_path))
    else:
        print(colored("file in destination already exists, move aborted: %s" % dest_path_file, 'yellow'))


def check_existing_files(dest_path, rover):
    """ check if rinex files are already available in the processing directory, otherwise copy & uncompress files from server
    :param dest_path: temp directory for preprocessing the GNSS rinex files
    :param rover: rover file name prefix
    :return: doy_max: maximum doy in existing files, newer doys should be copied
    """
    # get parent directory
    parent_dir = os.path.dirname(os.path.dirname(dest_path))
    print('parent dir: ', parent_dir)

    # check if any observation file exists in parent directory
    # with the handed rover name/number
    if any(i.startswith(rover) for i in os.listdir(parent_dir)):
        # if observation files of given rover already exist:
        # get file names
        files = sorted(glob.iglob(parent_dir + '/' + rover + '???0.*O', recursive=True), reverse=True)
        # get newest year of files in processing folder
        year_max = max([os.path.basename(f).split('.')[1][:2] for f in files])
        print(colored('newest year in existing files of parent dir: %s' % year_max, 'blue'))

        # get newest doy of files in processing folder
        doy_max = \
        os.path.basename(sorted(glob.iglob(parent_dir + '/' + rover + '???0.' + year_max + 'O', recursive=True),
                                reverse=True)[0]).split('.')[0][4:7]
        print(colored('newest doy in existing files of parent dir: %s' % doy_max, 'blue'))
    else:
        # if no observation files exist, set default values:
        doy_max = '320'     # default value
        year_max = '21'     # default value
        print("There exists no " + rover + "-rover observation file in parent-directory.")
        print("Data to copy from source destination, is searched starting with year 20" + year_max + " and day-of-year " + doy_max + "!!!")
    return year_max, doy_max


def copy_file_no_overwrite(source_path, dest_path, file_name):
    """ copy single files without overwriting existing files
    :param source_path: source directory
    :param dest_path: destination directory
    :param file_name: name of file to copy
    """
    # construct the src path and file name
    source_path_file = os.path.join(source_path, file_name)

    # construct the dest path and file name
    dest_path_file = os.path.join(dest_path, file_name)

    # test if the dest file exists, if false, do the copy, or else abort the copy operation.
    if not os.path.exists(dest_path_file):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        shutil.copyfile(source_path_file, dest_path_file)
        print("\ncopy from %s to %s \nok" % (source_path_file, dest_path_file))
    else:
        print("\nfile in destination already exists: %s, \ncopy aborted!!!" % dest_path_file)
    pass


def copy_solplotsdirs(source_path, dest_path):
    """ copy entire solution and plot directories
    :param source_path: local directory containing the solution and plot files
    :param dest_path: remote directory used for backup
    """
    shutil.copytree(source_path + '20_solutions/', dest_path + '20_solutions/', dirs_exist_ok=True)
    print('\ncopy directory: ' + source_path + '20_solutions/\nto: ' + dest_path + '20_solutions/')
    shutil.copytree(source_path + '30_plots/', dest_path + '30_plots/', dirs_exist_ok=True)
    print('copy directory: ' + source_path + '30_plots/\nto: ' + dest_path + '30_plots/')


def copy4backup(source_path, dest_path):
    """ copy entire processing directory to server
    :param source_path: local processing directory containing
    :param dest_path: remote directory used for backup
    """
    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
    print('\ncopy directory: ' + source_path + '\nto: ' + dest_path)


# function for getting all files in lower folders of one directory, call like:
# "for file in get_files(dir_PATH):
#     print(file)"
get_files = lambda path: (os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files)


def copy_rinex_files(source_path, dest_path, receiver=['NMLB', 'NMLR', 'NMER', 'nmsh'], copy=[True, False],
                     parent=[True, False], hatanaka=[True, False], move=[True, False], delete_temp=[True, False]):
    """ copy rinex files of remote directory to a local temp directory if it does not already exist
        & uncompress files, keep observation (and navigation) files
        & delete compressed files and subfolders afterwards
    :param source_path: remote directory hosting compressed GNSS rinex files
    :param dest_path: local directory for processing the GNSS rinex files
    :param receiver: high-end (Leica) or low-cost (Emlid), needs to be treated differently
    :param copy: do you want to copy (True) the data or skip copying (False) the data and just decompress and further?
    :param move: move renamed files to parent directory (True) or not (False)
    :param delete_temp: delete temporary directory (True) or not (False)
    """
    # Q: create temp directory if not existing
    create_folder(dest_path)

    # Q: check which rinex files are already existing in the processing directory
    # prepare file prefix
    if receiver == 'NMLR':
        rover = '3393'
    if receiver == 'NMLB':
        rover = '3387'
    if receiver == 'NMER' or receiver == 'nmsh':
        rover = receiver

    # Q: check already existing years and doys of files in processing directory, get newest yeardoy
    year_max, doy_max = check_existing_files(dest_path, rover)

    if receiver == 'NMER':
        # Q: copy files from network drive to local temp folder
        if copy is True:
            for f in sorted(glob.glob(source_path + '*.zip'), reverse=False):
                # construct the destination filename
                dest_file = os.path.join(dest_path, f.split("\\")[-1])
                # convert datetime to day of year (doy) from newest filename in source directory
                doy_file = datetime.datetime.strptime(os.path.basename(f).split('_')[2], "%Y%m%d%H%M").strftime('%j')
                yy_file = os.path.basename(f).split('_')[2][2:4]

                # Q: only copy files from server which are newer than the already existing doys of year=yy
                if (yy_file == year_max and doy_file >= doy_max) or (yy_file > year_max):
                    # copy file if it does not already exist
                    if not os.path.exists(dest_path + dest_file):
                        shutil.copy2(f, dest_path)
                        print("\nfile copied from %s to %s" % (f, dest_file))
                    else:
                        print(colored("\nfile in destination already exists: %s, \ncopy aborted!!!" % dest_file,
                                      'yellow'))
                        continue

                    # Q: uncompress file
                    shutil.unpack_archive(dest_file, dest_path)
                    print('file decompressed: %s' % dest_file)
                else:
                    # print(colored('file already preprocessed and available in the processing folder, skip file: %s' % f, 'yellow'))
                    # doy_file = None
                    pass
            if doy_file is None:
                print(colored('no new files available in source folder', 'green'))
        else:
            pass

        # Q: delete nav & zipped files
        if doy_file is not None:
            for f in glob.glob(dest_path + '*.*[BPzip]'):
                os.remove(f)
            print("nav files deleted %s" % dest_path)

            # Q: split & merge day-overlapping Emlid rinex files to daily rinex files (for Emlid files only!)
            dayoverlapping2daily_rinexfiles(dest_path, 'ReachM2_sladina-raw_', receiver, move, delete_temp)

        # delete the first and last file, which do not cover a whole day!! (they only include hours from 3:00-0:00 or from 0:00-03:00)
        print('\ndeleting all observation files that do not contain a whole day!!')
        os.remove(dest_path + receiver + doy_max + '0.' + year_max + 'o')
        for f in glob.glob(dest_path + os.listdir(dest_path)[-1]):
            os.remove(f)

        # move renamed files to parent directory
        if move is True:
            for f in glob.iglob(dest_path + receiver + '???0.*O', recursive=True):
                move_files2parentdir(dest_path, f)
        else:
            print('renamed merged daily files are NOT moved to parent directory!')


    if receiver == 'NMLB' or receiver == 'NMLR':
        # Q: copy files from network drive to local temp folder
        if copy is True:
            for f in glob.glob(source_path + '*.tar.xz'):
                # create destination filename
                dest_file = dest_path + f.split("\\")[-1]
                doy_file = os.path.basename(f)[4:7]
                yy_file = os.path.basename(f).split('.')[1][:2]

                # Q: only copy files from server which are newer than the already existing doys of year=yy
                if (yy_file == year_max and doy_file > doy_max) or (yy_file > year_max):
                    # copy file if it does not already exist
                    if not os.path.exists(dest_path+dest_file):
                        shutil.copy2(f, dest_path)
                        print("\nfile copied from %s to %s" % (f, dest_file))
                    else:
                        print(colored("\nfile in destination already exists: %s, \ncopy aborted!!!" % dest_file,
                                      'yellow'))
                        continue

                    # Q: uncompress .tar.xz file
                    with tarfile.open(fileobj=lzma.open(dest_file)) as tar:
                        tar.extractall(dest_path)
                        print('file decompressed: %s' % dest_file)
                        # close xz file
                        tar.fileobj.close()
                else:
                    # print(colored('file already preprocessed and available in the processing folder, skip file: %s' % f, 'yellow'))
                    # doy_file = None
                    pass
            if doy_file is None:
                print(colored('no new files available in source folder:', 'green'))

        else:
            pass

        # Q: move obs (and nav) files to parent dir
        if parent is True:
            if receiver == 'NMLB':
                # copy observation (.yyd) & navigation (.yy[ngl]) files from base receiver
                for f in glob.glob(dest_path + 'var/www/upload/' + receiver + '/*.*'):
                    shutil.move(f, dest_path)
                if doy_file is not None:
                    print(colored("\nobs & nav files moved to parent dir %s" % dest_path, 'blue'))
            if receiver == 'NMLR':
                # copy only observation (.yyd) files from rover receivers
                for f in glob.glob(dest_path + 'var/www/upload/' + receiver + '/*.*d'):
                    shutil.move(f, dest_path)
                if doy_file is not None:
                    print(colored("\nobs files moved to parent dir %s" % dest_path, 'blue'))
        else:
            pass

        # Q: convert hatanaka compressed rinex (.yyd) to uncompressed rinex observation (.yyo) files
        if hatanaka is True:
            if doy_file is not None:
                print(colored("\ndecompress hatanaka rinex files", 'blue'))
                for hatanaka_file in glob.glob(dest_path + '*.*d'):
                    print('decompress hatanaka file: ', hatanaka_file)
                    # subprocess.Popen('crx2rnx ' + hatanaka_file)
                    subprocess.call('crx2rnx ' + hatanaka_file)
                print(colored("\nfinished decompressing hatanaka rinex files", 'blue'))
        else:
            pass

        # Q: move all obs (and nav) files from temp to parent directory
        if move is True:
            if doy_file is not None:
                print(colored("\nmove decompressed files to parent dir", 'blue'))
                for f in glob.glob(dest_path + '*.*[ongl]'):
                    move_files2parentdir(dest_path, f)
                print(colored("\nfinished moving decompressed files to parent dir", 'blue'))

        else:
            print('files are NOT moved to parent directory!')


    if receiver == 'nmsh':
        # Q: copy files from network drive to local temp folder
        if copy is True:
            for f in get_files(source_path):
                # Q: construct the destination filename
                dest_file = ''.join(os.path.basename(f))
                # Q: get day of year (doy) and year from newest filename in source directory
                doy_file = ''.join(re.findall("\d+",os.path.basename(f).split('.')[0]))
                yy_file = ''.join(re.findall("\d+",os.path.basename(f).split('.')[1]))

                # Q: only copy files from server which are newer than the already existing doys of year=yy
                if (yy_file == year_max and doy_file > doy_max) or (yy_file > year_max):

                    # Q: copy, unpack and convert file if it does not already exist
                    if not os.path.exists(dest_path+dest_file):

                        # Q: copy files from server to local temporary-processing directory
                        shutil.copy2(f, dest_path)
                        print("\nfile: %s copied \nfrom %s \nto %s" % (dest_file, source_path, dest_path))

                        # Q: unpack 7-zip archive
                        py7zr.unpack_7zarchive(dest_path + dest_file, dest_path, extra=None)
                        filename = dest_file.split('.')[0]
                        unpacked_file = filename + '.' + dest_file.split('.')[1]
                        print("7-zip-archive (%s) unpacked to jps-file (%s)" % (dest_file, unpacked_file))

                        # Q: rename "*.yyjps" to "*.jps" to be able to process the file with jps2rin
                        jps_file = filename + '.jps'
                        os.rename(dest_path + unpacked_file, dest_path + jps_file)

                        # Q: convert jps file into rinex format by calling jps2rin application
                        print("start converting jps-file into rinex-format")
                        jps2rin(dest_path, jps_file, '3.03', agency='Alfred_Wegener_Institute', rcv_type='JAVAD_OEM_Board_TR-G3t', ant_type='JAVAD_GrAnt-G3T')

                        # delete .jps-files (but not the 7z-archive!)
                        os.remove(dest_path+jps_file)
                        print("deleted jps-file in temporary folder")

                    else:
                        print(colored("\nfile in destination already exists: %s, \ncopy aborted!!!" % dest_file,
                                      'yellow'))
                        continue

                else:
                    pass

            if doy_file is None:
                print(colored('no new files available in source folder', 'green'))
        else:
            pass

        if doy_file is not None:
            # Q: merge horal javad observation files into daily rinex files (no nav-files merged, as Leica nav-files are used)
            horal2daily_rinexfiles(dest_path, receiver)

        # move renamed observation files to parent directory
        if move is True:
            for f in glob.iglob(dest_path + receiver + '???0.*O', recursive=True):
                move_files2parentdir(dest_path, f)
        else:
            print('renamed merged daily files are NOT moved to parent directory!')

    # Q: get the newest year and doy after copying, and convert to modified julian date (mjd)
    print('\nafter copyig the observation files from server and pre-processing them \nthe newest year and doy are in:')
    yy_file, doy_file = check_existing_files(dest_path, rover)
    date_file = gnsscal.yrdoy2date(int('20' + yy_file), int(doy_file))
    mjd_newest_file = jdcal.gcal2jd(date_file.year, date_file.month, date_file.day)[1]

    # Q: delete temp directory
    if delete_temp is True:
        remove_folder(dest_path)
    else:
        print('temporary directory is NOT deleted!')

    return mjd_newest_file


def merge_split_Leica(dest_path, delete_temp_merge):
    """
    merge all Leica observation and navigation files that are splitted
    and show the naming structure "NAMEdoy???.yy[o/g/l/n]"
    :param dest_path:           directory containing the splitted observation and navigation files
    :param delete_temp_merge:   delete temporary directory (True) or not (False)
    """
    # Q: create a temporary folder for merging all splitted files (from Leica Rover and Base)
    dir_path_temp_merge = dest_path + 'temp_merge/'
    create_folder(dir_path_temp_merge)
    # Q: create a temporary folder for all merged files
    dir_path_merged = dir_path_temp_merge + 'merged/'
    create_folder(dir_path_merged)

    # Q: find and list all split files in processing directory
    #    that have the string-structure "NAMEdoy[a-x]MM.yy[o/g/l/n]" / "NAMEdoy???.yy[o/g/l/n]"
    list_NAMEdoyaMM = [s for s in os.listdir(dest_path) if (len(s) == 14 and (s[-1:] == 'g' or s[-1:] == 'l' or s[-1:] == 'n' or s[-1:] == 'o'))]

    if list_NAMEdoyaMM == []:
        print('\nNO split Leica files in processing directory to be merged')
    else:
        print('\nmoving all split observation and navigation-files '
              'from processing directory to temporary merging directory '
              'for merging them with gfzrnx:\n%s' % list_NAMEdoyaMM)

        # Q: move all split obs and nav files (with structure NAMEdoy???.yy[o/g/l/n]") to temp_merge directory
        for f in list_NAMEdoyaMM:
            os.replace(dest_path + f, dir_path_temp_merge + f)

        # Q: relocate corresponding files (with structure NAMEdoy0.yy[o/l/n/g]),
        #    merge all files by yy and doy and safe merged files in a "merged-folder"
        # Q: observation files
        move_and_merge(dest_path, dir_path_temp_merge, dir_path_merged, file_extension='o')
        # Q: GPS files
        move_and_merge(dest_path, dir_path_temp_merge, dir_path_merged, file_extension='n')
        # Q: Galileo files
        move_and_merge(dest_path, dir_path_temp_merge, dir_path_merged, file_extension='l')
        # Q: GLONASS files
# TODO:
#       move_and_merge(dest_path, dir_path_temp_merge, file_extension='g')
#       GLONASS DOES NOT WORK YET!

    # Q: create a dataframe that includes filenames, receiver-name, year and doy of merged files
    list_LR_merged = [s for s in os.listdir(dir_path_merged) if (s[-1:] == 'o' and '3393' in s)]
    list_LB_merged = [s for s in os.listdir(dir_path_merged) if (s[-1:] == 'o' and '3387' in s)]
    list_GPS_merged = [s for s in os.listdir(dir_path_merged) if s[-1:] == 'n']
    list_Galileo_merged = [s for s in os.listdir(dir_path_merged) if s[-1:] == 'l']
# TODO:
#   list_GLONASS_merged = [s for s in os.listdir(dir_path_merged) if s[-1:] == 'g']
    df_LR_merged = create_DataFrame(list_LR_merged)
    df_LB_merged = create_DataFrame(list_LB_merged)
    df_LB_merged.insert(loc=1, column='GPS_File', value=list_GPS_merged)
    df_LB_merged.insert(loc=2, column='Galileo_File', value=list_Galileo_merged)
# TODO:
#   df_LB_merged.insert(loc=6, column='GLONASS_File', value=list_GLONASS_merged)

    if list_NAMEdoyaMM != []:
        # Q: move all merged observation and navigation files from temporary folder to processing directory
        print('\nthe following observation files of Leica Rover were merged '
              'and are replaced in processing directory: \n%s'
              % df_LR_merged)
        print('\nthe following observation and navigation files of Leica Base '
              'were merged and are replaced in processing directory: \n%s'
              % (df_LB_merged))

        for f in os.listdir(dir_path_merged):
            os.replace(dir_path_merged + f, dest_path + f)

    # Q: delete temp directory
    if delete_temp_merge is True:
        remove_folder(dir_path_temp_merge)
        print('\ntemporary directory is deleted!')
    else:
        print('\ntemporary directory is NOT deleted!')

    return df_LR_merged, df_LB_merged


def move_and_merge(dest_path, dir_path_temp_merge, dir_path_merged, file_extension=['o', 'g', 'l', 'n']):
    """
    list all files with the given file extension and the structure "NAMEdoy???.yy[o/g/l/n]" in the dir_path_temp_merge
    and search for corresponding files (with same file_extension, same yy and same doy) with the structure
    "NAMEdoy0.yy[o/l/n/g]" in dest_path. Relocate all existing files to temporary merging folder, merge them by
    year and doy and safe the merged-files in an extra folder.
    :param dest_path:           directory containing the splitted observation and navigation files
    :param dir_path_temp_merge: directory path to temporary folder for merging
    :param dir_path_merged:     directory path to merged files
    :param file_extension:      file suffix of the observation and navigation files
    """
    # Q: find and list all splitted observation files in merging directory
    #    that have the string-structure "NAMEdoy[a-x]MM.yy[o/g/l/n]" / "NAMEdoy???.yy[o/g/l/n]"
    list_files = [s for s in os.listdir(dir_path_temp_merge) if (len(s) == 14 and s[-1:] == file_extension)]

    # Q: move all corresponding splitted obs and nav files (with structure NAMEdoy0.yy[o/l/n/g]) to temp_merge directory
    for f in list_files:
        # Q: get receiver-prefix, year and doy of splitted file
        receiver_prefix = f[0:4]
        yy = f[-3:-1]
        doy = f[4:7]

        # Q: construct filenames of splitted files with the structure NAMEdoy0.yy[o/l/n/g]
        filename = receiver_prefix + doy + '0.' + yy + file_extension

        # Q: move and merge all observation files, if they exist in processing directory
        #    query is needed as some days are splitted multiple times and their yy+doy are asked for several times
        if os.path.exists(dest_path + filename):
            # Q: move all obs and nav files (with structure NAMEdoy0.YY[o/l/n/g]) to temp_merge directory
            os.replace(dest_path + filename, dir_path_temp_merge + filename)

            # Q: construct gfzrnx input filenames
            gfzrnx_input = receiver_prefix + doy + '*.' + yy + file_extension

            # Q: merge observation and navigation files with gfzrnx
            merge_rinex_files(dir_path_temp_merge, gfzrnx_input)

            # Q: construct gfzrnx-output / merged-rinex filenames
            gfzrnx_output = [s for s in os.listdir(dir_path_temp_merge) if (receiver_prefix in s and yy in s and doy in s and s.endswith('rnx'))][0]

            # Q: move merged rinex files into "merged"-folder
            os.replace(dir_path_temp_merge + gfzrnx_output, dir_path_merged + gfzrnx_output)

            # Q: rename merged rinex file to match naming-convention NAMEdoy0.YYo
            os.rename(dir_path_merged + gfzrnx_output, dir_path_merged + filename)


def create_DataFrame(list_files):
    """
    create a DataFrame from a list of files.
    DataFrame contains the columns: filenames, receiver, year and doy.
    :param list_files: List of filenames with the structure "NAMEDOY0.yyo"
    """
    # Q: create data frame to save filenames, receiver, year and doy with the length of handed list
    df_files = pd.DataFrame(columns=['Filename', 'Receiver', 'Year', 'DOY'], index=range(0, len(list_files)))

    for i, f in enumerate(list_files):
        # Q: get receiver-prefix, year and doy of files in list
        receiver_prefix = f[0:4]
        yy = f[-3:-1]
        doy = f[4:7]

        # Q: save filenames, receiver, year and doy of files in data frame
        df_files.loc[i]['Filename'] = f
        df_files.loc[i]['Receiver'] = receiver_prefix
        df_files.loc[i]['Year'] = yy
        df_files.loc[i]['DOY'] = doy

    return df_files

def process_merged_Leica(df_LR_merged, df_LB_merged, dest_path, mjd_start, mjd_end_ER_LB, mjd_end_LR_LB, ti_int, base_prefix,
                         brdc_nav_prefix, precise_nav_prefix, resolution, ending, options_ER_LB, options_LR_LB, base_name):
    """
    merge and "re-process" all Leica observation and navigation files that are split
    and show the naming structure "NAMEdoy???.yy[o/g/l/n]". And replace newly processed solution files.
    :param df_LR/LB_merged:     DataFrame containing filenames and dates of split files
    :param dest_path:           directory containing split observation and navigation files
    :param mjd_start:           start modified julian date to process
    :param mjd_end_ER/LR_LB:    end modified julian date to process
    :param ti_int:              processing time interval (in seconds)
    :param base_prefix:         prefix of base rinex filename
    :param brdc_nav_prefix:     prefix of broadcast navigation filename
    :param precise_nav_prefix:  prefix of precise orbit filename
    :param resolution:          processing time interval (in minutes) for naming of output folder
    :param ending:              suffix of solution file names (e.g. a varian of processing options: '_noglonass'
    :param options_ER/LR_LB:    rtkpost configuration files of Emlid Rover and Leica Rover with Leica Base
    :param base_name:           name/abbreviation of base eg. 'LB'
    """
    # Q: Only re-process if newly merged files exist
    if df_LR_merged.empty is False or df_LB_merged.empty is False:

        # Q: calculate the start mjd of newly merged Leica files for Rover and Base
        start_date_LR = gnsscal.yrdoy2date(int('20' + df_LR_merged['Year'].iloc[0]), int(df_LR_merged['DOY'].iloc[0]))
        start_mjd_merged_LR = jdcal.gcal2jd(start_date_LR.year, start_date_LR.month, start_date_LR.day)[1]
        start_date_LB = gnsscal.yrdoy2date(int('20' + df_LR_merged['Year'].iloc[0]), int(df_LR_merged['DOY'].iloc[0]))
        start_mjd_merged_LB = jdcal.gcal2jd(start_date_LB.year, start_date_LB.month, start_date_LB.day)[1]

        # Q: only process newly merged files of which solutions already exist
        if mjd_end_LR_LB >= start_mjd_merged_LR or mjd_end_LR_LB >= start_mjd_merged_LB or mjd_end_ER_LB >= start_mjd_merged_LB:

            # Q: copy all merged files and needed year + doys of the other receiver in a temp_processing folder
            gather_merged_files_4_processing(dest_path, df_LR_merged, df_LB_merged)

            # Q: process all files in temporary re-processing dir
            automate_rtklib_pp(dest_path + 'temp_reprocessing_merged/', 'NMER', mjd_start, mjd_end_ER_LB, ti_int, base_prefix, brdc_nav_prefix, precise_nav_prefix, resolution, ending, options_ER_LB, 'NMER', base_name)
            automate_rtklib_pp(dest_path + 'temp_reprocessing_merged/', '3393', mjd_start, mjd_end_LR_LB, ti_int, base_prefix, brdc_nav_prefix, precise_nav_prefix, resolution, ending, options_LR_LB, 'NMLR', base_name)


            # Q: relocate newly re-processed .pos - solution files and replace the old solution files in 20_solutions directory
            replace_solution_files(dest_path, 'NMER', base_name, resolution)
            replace_solution_files(dest_path, 'NMLR', base_name, resolution)

            # Q: delete temp directory
            temp_processing = dest_path + 'temp_reprocessing_merged/'
            if os.path.exists(temp_processing):
                remove_folder(temp_processing)
                print('\n temporary re-processing directory removed!')

        else:
            print('\nNO of the newly merged Leica files for Base and Rover need to be re-processed, as they are newer than the end date of existing solution files')
    else:
        print('\nNO newly merged Leica files to re-process!')


def gather_merged_files_4_processing(dest_path, df_LR_merged, df_LB_merged):
    """
    create a temporary processing directory for "re-processing" the merged files that were already processed
    and relocate all needed files of Leica Rover and Base and Emlid Rover to it.
    :param dest_path:       path of directory that contains the observation and navigation files to be processed
    :param df_LR_merged:    DataFrame of merged (obs) files from Leica Rover
                            (containing the filenames that shall be processed)
    :param df_LB_merged:    DataFrame of merged (obs + nav) files from Leica Base
    """
    # Q: create a temporary processing folder if not already exiting
    temp_processing = dest_path + 'temp_reprocessing_merged/'
    create_folder(temp_processing)

    # Q: construct needed filenames
    df_LB_merged.insert(loc=0, column='LR_Filename', value='3393' + df_LB_merged['DOY'] + '0.' + df_LB_merged['Year'] + 'o')
    df_LB_merged.insert(loc=0, column='ER_Filename', value='NMER' + df_LB_merged['DOY'] + '0.' + df_LB_merged['Year'] + 'o')
    df_LR_merged.insert(loc=1, column='LB_Filename', value='3387' + df_LB_merged['DOY'] + '0.' + df_LB_merged['Year'] + 'o')
    df_LR_merged.insert(loc=2, column='LB_GPS_Filename', value='3387' + df_LB_merged['DOY'] + '0.' + df_LB_merged['Year'] + 'n')
    df_LR_merged.insert(loc=3, column='LB_Galileo_Filename', value='3387' + df_LB_merged['DOY'] + '0.' + df_LB_merged['Year'] + 'l')
# TODO:
#   df_LR_merged.insert(loc=4, column='LB_GLONASS_Filename', value='3387' + df_LB_merged['DOY'] + '0.' + df_LB_merged['Year'] + 'g')

    # Q: relocate (Leica Base observation and navigation, Leica Rover and Emlid Rover) files
    #    to temporary processing directory based on year+doy of merged Leica BASE Files
    print('\ncopy (Leica Base observation and navigation, Leica Rover and Emlid Rover observation) files '
          'to temporary processing directory based on year and doy of newly merged Leica BASE Files.')
    for f in chain(df_LB_merged['Filename'], df_LB_merged['LR_Filename'], df_LB_merged['ER_Filename'], df_LB_merged['GPS_File'], df_LB_merged['Galileo_File']):       # TODO: , df_LB_merged['GLONASS_File']):
        if os.path.exists(dest_path + f):
            shutil.copy2(dest_path + f, temp_processing)
        else:
            print('%s does not exist in processing directory and could not be moved to temporary processing folder!' % f)

    # Q: relocate (Leica Base obervation and navigation and Leica Rover) files
    #    to temporary processing directory based on year+doy of merged Leica ROVER Files
    print('\ncopy (Leica Base observation and navigation and Leica Rover observation) files '
          'to temporary processing directory based on year and doy of newly merged Leica ROVER Files.')
    for f in chain(df_LR_merged['Filename'], df_LR_merged['LB_Filename'], df_LR_merged['LB_GPS_Filename'], df_LR_merged['LB_Galileo_Filename']):                      # TODO: , df_LR_merged['LB_GLONASS_Filename']):
        if os.path.exists(dest_path+f):
            shutil.copy2(dest_path + f, temp_processing)
        else:
            print('%s does not exist in processing directory or was already relocated to temporary processing directory' % f)

    # Q: copy antex and rtklib-configuration files to temporary processing folder
    for f in glob.iglob(dest_path + '*.atx', recursive=True):
        shutil.copy2(f, temp_processing)
    for f in glob.iglob(dest_path + '*.conf', recursive=True):
        shutil.copy2(f, temp_processing)


def replace_solution_files(dest_path, rover_name, base_name, resolution):
    """
    relocate newly processed .pos - solution files and replace the old solution files in 20_solutions directory,
    move the observation files back to processing directory and delete temporary "re-processing" directory
    :param dest_path:   directory path to processing_dir (that contains the newly merged obs files that need to be re-processed)
    :param rover_name:  ROVER-abbreviation and folder/file_appendix e.g. 'NMER' or '3393'
    :param base:        BASE-abbreviation and folder_name e.g. 'LB'
    :param resolution:  processing time interval (minutes) for naming of output folder
    """
    # Q: create path variables
    temp_processing = dest_path + 'temp_reprocessing_merged/'
    solutions_path = '20_solutions/' + rover_name + '_' + base_name + '/' + resolution + '/'
    NEW_solutions = temp_processing + solutions_path + 'temp_' + rover_name + '/'
    OLD_solutions = dest_path + solutions_path
    list_NEW_solutions = os.listdir(NEW_solutions)

    # Q: relocate newly re-processed .pos - solution files and replace the old in 20_solutions directory
    if os.path.exists(NEW_solutions):
        for f in list_NEW_solutions:
            os.replace(NEW_solutions + f, OLD_solutions + f)
        print('\n newly re-processed solution files of %s and %s replaced in 20_solution directory: \n%s'% (rover_name, base_name, list_NEW_solutions))


def convert_datetime2doy_rinexfiles(dest_path, rover_prefix, rover_name):
    """ convert Emlid file names to match format for 'gfzrnx' rinex conversion tools
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    :param rover_prefix: prefix of rinex files in temp directory
    :param rover_name: name of rover receiver

    input filename: 'ReachM2_sladina-raw_202111251100.21O'  [rover_prefix + datetime + '.' + yy + 'O']
    output filename: 'NMER329[a..d].21o'                    [rover_prefix + doy + '0.' + yy + 'o']
    """
    # Q: get doy from rinex filenames in temp dir with name structure: 'ReachM2_sladina-raw_202112041058.21O' [rover_prefix + datetime + '.' + yy + 'O']
    print(colored('\nrenaming all files', 'blue'))
    for f in glob.iglob(dest_path + rover_prefix + '*.*O', recursive=True):
        # get rinex filename and year
        rover_file = os.path.basename(f)
        yy = rover_file.split('.')[-1][:2]
        # convert datetime to day of year (doy)
        doy = datetime.datetime.strptime(rover_file.split('.')[0].split('_')[2], "%Y%m%d%H%M").strftime('%j')
        # create new filename with doy
        new_filename = dest_path + rover_name + doy + 'a.' + yy + 'o'
        print('\nRover file: ' + rover_file, '\ndoy: ', doy, '\nNew filename: ', new_filename)

        # check if new filename already exists and rename file
        file_exists = os.path.exists(new_filename)  # True or False
        if file_exists is True:
            new_filename = dest_path + rover_name + doy + 'b.' + yy + 'o'
            # check if new filename already exists and rename file
            file_exists = os.path.exists(new_filename)  # True or False
            if file_exists is True:
                new_filename = dest_path + rover_name + doy + 'c.' + yy + 'o'
                # check if new filename already exists and rename file
                file_exists = os.path.exists(new_filename)  # True or False
                if file_exists is True:
                    new_filename = dest_path + rover_name + doy + 'd.' + yy + 'o'
                else:
                    os.rename(f, new_filename)
                    print('\nNew filename already existing --> renamed to: ', new_filename)
            else:
                os.rename(f, new_filename)
                print('\nNew filename already existing --> renamed to: ', new_filename)
        else:
            os.rename(f, new_filename)

    print(colored('\nfinished renaming all files', 'blue'))


def split_rinex(dest_path, rover_name):
    """ split day-overlapping rinex files at midnight --> get multiple subdaily files
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    :param rover_name: name of rover receiver

    gfzrnx split input:  'NMER329[a..d].21o'    [rover + doy + '.' + yy + 'o']
    gfzrnx split output: '    00XXX_R_20213291100_01D_30S_MO.rnx'
    """
    print(colored('\nstart splitting day-overlapping rinex files', 'blue'))
    for f in glob.iglob(dest_path + rover_name + '*.*o', recursive=True):
        # get filename
        rover_file = os.path.basename(f)
        print('\nstart splitting day-overlapping rinex file: %s' % rover_file)

        # split rinex file at midnight with command: 'gfzrnx -finp NMER345.21o -fout ::RX3:: -split 86400'

        #'gfzrnx -finp NMERdddf.yyo -fout ::RX3:: -split 86400'

        process1 = subprocess.Popen('cd ' + dest_path + ' && '
                                                        'gfzrnx -finp ' + rover_file + ' -fout ::RX3:: -split 86400',
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

        stdout1, stderr1 = process1.communicate()
        print(stdout1)
        print(stderr1)

    print(colored('\nfinished splitting all day-overlapping rinex files at: %s' % dest_path, 'blue'))


def rename_splitted_rinexfiles(dest_path, rover_name):
    """ rename gfzrnx splitted rinex files output names to match input for gfzrnx merge files:
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    :param rover_name: name of rover receiver

    gfzrx split output: '    00XXX_R_20213291100_01D_30S_MO.rnx'
    gfzrx merge input:  'NMER00XXX_R_20213291100_01D_30S_MO.yyo'
    """
    # Q: rename all .rnx files (gfzrnx split output --> gfzrnx merge input)
    print(colored('\nrenaming all splitted rinex files', 'blue'))
    for f in glob.iglob(dest_path + '*.rnx', recursive=True):
        rover_file = os.path.basename(f)
        yy = rover_file.split('_')[2][2:4]
        new_filename = dest_path + rover_name + rover_file.split('.')[0][4:] + '.' + yy + 'o'
        print('\nRover file: ' + rover_file, '\nNew filename: ', new_filename)
        os.rename(f, new_filename)

    print(colored('\nfinished renaming all splitted rinex files', 'blue'))


def merge_rinex_files(dest_path, gfzrnx_input):
    """
    split and merge rinex navigation files (.yy[g//l/n]) together per day --> get a daily rinex file
        :param dest_path: local temporary directory for preprocessing/merging the GNSS rinex files
        :param gfzrnx_input: the "common" filename(s) that shall be merged
        Example for GPS data:
        gfzrnx input:   "3387018*.23n", the "*" is needed so that all files that are merged can be found
        gfzrnx output:  "NAME00XXX_R_20YYDOY0000_01D_GN.rnx"
        Example for observation files:
        gfzrnx input:   "nmsh001?.22o", the "?" needs to be such that all files that are merged can be found
                        (structure of 24-horal files here:  'nmshDOY[a-x].yyo')
        gfzrnx output:  "NMSH00XXX_R_20220010000_01D_01S_MO.rnx"
    """
    # Q: merge NAVIGATION files
    if gfzrnx_input.endswith('g') or gfzrnx_input.endswith('l') or gfzrnx_input.endswith('n'):
        # Q: merge rinex files per day with command: 'gfzrnx -finp gfzrnx_input -fout ::RX3:: -kv -f -split 86400'
        print("\nstart merging navigation files %s in directory: %s" % (gfzrnx_input, dest_path))
        process1 = subprocess.Popen('cd ' + dest_path + ' && '
                                                        'gfzrnx -finp ' + gfzrnx_input + ' -fout ::RX3:: -kv -f -split 86400',
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

    # Q: merge OBSERVATION files
    if gfzrnx_input.endswith('o'):
        # Q: merge rinex files per day with command: 'gfzrnx -finp gfzrnx_input -fout ::RX3D:: -d 86400'
        print("\nstart merging observation files %s in directory: %s" % (gfzrnx_input, dest_path))
        process1 = subprocess.Popen('cd ' + dest_path + ' && '
                                                        'gfzrnx -finp ' + gfzrnx_input + ' -fout ::RX3D:: -d 86400',
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

    stdout1, stderr1 = process1.communicate()
    print(stdout1)
    print(stderr1)


def merge_rinex(dest_path):
    """ merge rinex files together per day --> get daily rinex files of emlid receiver (NMER)
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files

    gfzrnx merge input:  'NMER00XXX_R_2021329????_01D_30S_MO.yyo'
    gfzrnx merge output: 'NMER00XXX_R_2021330????_01D_30S_MO.rnx'
    """
    print(colored('\nmerging all rinex files per day at: %s' % dest_path, 'blue'))
    for f in glob.iglob(dest_path + 'NMER00XXX_R_20' + '*.*O', recursive=True):
        # get filename
        rover_file = os.path.basename(f)
        yy = rover_file.split('_')[2][2:4]
        # extract doy
        doy = rover_file.split('.')[0][16:19]
        print('\nRover file: ' + rover_file, '\ndoy: ', doy)

        # merge rinex files per day with command: 'gfzrnx -finp NMER00XXX_R_2021330????_01D_30S_MO.21o' -fout ::RX3D:: -d 86400'
        process1 = subprocess.Popen('cd ' + dest_path + ' && '
                                                        'gfzrnx -finp NMER00XXX_R_20' + yy + doy + '????_01D_30S_MO.' + yy + 'o -fout ::RX3D:: -d 86400',
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

        stdout1, stderr1 = process1.communicate()
        print(stdout1)
        print(stderr1)

    print(colored('\nfinished merging all rinex files per day at: %s' % dest_path, 'blue'))


def merge_rinex_JAVAD(dest_path, file_prefix):
    """ merge rinex files together per day --> get daily rinex files
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    :param file_prefix: Gives the first 4 characters of the filename structure: "receiver_prefix + doy + 0 . yyo" e.g. 'nmsh'
    gfzrnx merge input:  'nmshDOYa.yyo' (with endings from a to x) 'nmshDOYx.yyo'
    gfzrnx merge output: 'NMSH00XXX_R_2021330????_01D_30S_MO.rnx'
    """
    print(colored('\nmerging all rinex files per day at: %s' % dest_path, 'blue'))
    for f in glob.iglob(dest_path + file_prefix + '*x.*o', recursive=True):
        # get filename
        rover_file = os.path.basename(f)
        # extract doy and year
        doy = ''.join(re.findall("\d+", rover_file.split('.')[0]))
        yy = ''.join(re.findall("\d+", rover_file.split('.')[1]))
        print('\nRover file: ' + rover_file, '\ndoy: ', doy)

        # merge rinex files per day with command: 'gfzrnx -finp nmsh????.yyo' -fout ::RX3D:: -d 86400'
        process1 = subprocess.Popen('cd ' + dest_path + ' && '
                                                        'gfzrnx -finp ' + file_prefix + doy + '?.' + yy + 'o -fout ::RX3D:: -d 86400',
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

        stdout1, stderr1 = process1.communicate()
        print(stdout1)
        print(stderr1)

    print(colored('\nfinished merging all rinex files per day at: %s' % dest_path, 'blue'))


def jps2rin(dir_path, filename, rnx_version, rnx_naming=False, agency='-Unknown-', rcv_type='-Unknown-', ant_type='-Unknown-'):
    """
    convert jps-files with ending ".jps" into rinex files (1 observation file and 3 navigation files)
    :param dir_path: directory path of the folder that contains the jps-file
    :param filename: filename with ending ".jps"
    :param rnx_version: Rinex version into which the file shall be converted (need to be given as string!)
                        options: 2.00|2.10|2.11|2.12|3.00|3.01|3.02|3.03|3.04|3.05
    :param rnx_naming: If the Rinex naming convention will be used for the output files
    :optional param Agency: XXX (string may not contain blanks!)
    :optional param Rcv_Type: Receiver Type (string may not contain blanks!)
    :optional param Ant_Type: Antenna Type (string may not contain blanks!)
    """
    if rnx_naming is False:
        jps2rin_process = subprocess.Popen('cd ' + dir_path + ' && '
                                                               'jps2rin -v="' + rnx_version + '" --AG="' + agency + '" --RT="' + rcv_type+ '" --AT="' + ant_type + '" ' + filename,
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

        jps2rin_stdout, jps2rin_stderr = jps2rin_process.communicate()
        print("jps2rin output: %s" % jps2rin_stdout)
        print("jps2rin errormessage: %s" % jps2rin_stderr)

    else:
        jps2rin_process = subprocess.Popen('cd ' + dir_path + ' && '
                                                               'jps2rin -v="' + rnx_version + '" --rn --AG="' + agency + '" --RT="' + rcv_type+ '" --AT="' + ant_type + '" ' + filename,
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

        jps2rin_stdout, jps2rin_stderr = jps2rin_process.communicate()
        print("jps2rin output: %s" % jps2rin_stdout)
        print("jps2rin errormessage: %s" % jps2rin_stderr)



def rename_merged_rinexfiles(dest_path, rover_name):
    """ rename gfzrnx merge files output names to match rtklib and leica rinex file names:
    :param dest_path: local temporary directory for preprocessing the GNSS rinex files
    :param rover_name: name of rover receiver
    gfzrnx merge output: 'NMER00XXX_R_2021330????_01D_30S_MO.rnx'
    rtklib input: 'NMERdoy0.yyo'  [rover_prefix + doy + '0.' + yy + 'o']
    """
    for f in glob.iglob(dest_path + '*.rnx', recursive=True):
        rover_file = os.path.basename(f)
        yy = rover_file.split('_')[2][2:4]
        new_filename = dest_path + rover_name + rover_file.split('.')[0][16:19] + '0.' + yy + 'o'
        print('\nRover file: ' + rover_file, '\nNew filename: ', new_filename)
        os.rename(f, new_filename)

    print(colored('\nfinished renaming all merged rinex files', 'blue'))



def dayoverlapping2daily_rinexfiles(dest_path, rover_prefix, receiver):
    """ convert day-overlapping Emlid rinex files to daily rinex files names to match rtklib input and leica files
        :param dest_path: local temporary directory for preprocessing the GNSS rinex files
        :param rover_prefix: prefix of rinex files in temp directory
        :param receiver: name of rover receiver ['NMER']
        """
    # create temporary directory if not already existing
    create_folder(dest_path)

    # convert Emlid files [rover_prefix + datetime + '.' + yy + 'O'] to format for 'gfzrnx' rinex conversion [rover_prefix + doy + '0.' + yy + 'o']
    convert_datetime2doy_rinexfiles(dest_path, rover_prefix, receiver)

    # split rinex files at midnight for day-overlapping files --> get subdaily rinex files
    split_rinex(dest_path, receiver)

    # rename splitted (subdaily) rinex files to match input for 'gfzrnx -merge'
    rename_splitted_rinexfiles(dest_path, receiver)

    # merge rinex files together per day --> get daily rinex files
    merge_rinex(dest_path)

    # rename merged (daily) rinex files to match rtklib input format [rover_prefix + doy + '0.' + yy + 'o'] & move to parent directory
    rename_merged_rinexfiles(dest_path, receiver)


def horal2daily_rinexfiles(dest_path, receiver):
    """ convert horal JAVAD-rinex files into daily rinex files.
        :param dest_path: local temporary directory for preprocessing the GNSS rinex files
        :param receiver: name of rover receiver ['nmsh']
        """
    # create temporary directory if not already existing
    create_folder(dest_path)

    # merge rinex files together per day --> get daily rinex files
    merge_rinex_JAVAD(dest_path, file_prefix=receiver)

    # rename merged (daily) rinex files to match rtklib input format [rover_prefix + doy + '0.' + yy + 'o'] & move to parent directory
    rename_merged_rinexfiles(dest_path, receiver)


def get_sol_yeardoy(dest_path, resolution, receiver=['NMLR', 'NMER'], base=['JB', 'LB']):
    """ get the newest solution file year, doy, mjd, date for only further process new available data.
    :param resolution: processing time interval (minutes) for naming of output folder
    :param receiver: name of rover receiver
    :param base: name of base receiver (for directory path)
    :return: start_yy, start_mjd
    """
    # check if a solution directory exists already, if yes
    # check if any solution file (.pos) exists in solution directory for the given receiver
    # if true: get the newest solution file year and doy
    # if false: take default values (a solution directory is created in function: "automate_rtklib_pp")
    if os.path.isdir(dest_path + '20_solutions/' + receiver + '_' + base + '/'):
        if any(i.endswith('.pos') for i in os.listdir(dest_path + '20_solutions/' + receiver + '_' + base + '/' + resolution + '/')):
            # get the newest solution file year and doy
            print(colored('\nget start year and mjd of existing solution files of rover: %s and base: %s for further processing' % (receiver, base), 'blue'))
            name_max = os.path.basename(
                sorted(glob.iglob(dest_path + '20_solutions/' + receiver + '_' + base + '/' + resolution + '/*.pos', recursive=True), reverse=True)[
                    0]).split('.')[0]
            start_yy = name_max[2:4]
            start_doy = int(name_max[-3:]) + 1
            start_date = gnsscal.yrdoy2date(int('20' + start_yy), start_doy)
            start_mjd = jdcal.gcal2jd(start_date.year, start_date.month, start_date.day)[1]
            print(colored('start year %s, doy %s, mjd %s, date %s' % (start_yy, start_doy, start_mjd, start_date), 'blue'))
        else:
            print(colored("\nThere are no solution files yet of rover: %s and base: %s. "
                          "Start year and mjd are set to default values for processing:" % (receiver, base), 'blue'))
            start_yy = '21'
            start_doy = 1
            start_date = gnsscal.yrdoy2date(int('20' + start_yy), start_doy)
            start_mjd = jdcal.gcal2jd(start_date.year, start_date.month, start_date.day)[1]
            print(colored('start year %s, doy %s, mjd %s, date %s' % (start_yy, start_doy, start_mjd, start_date), 'blue'))
    else:
        print(colored("\nThere are no solution files yet: of rover: %s and base: %s. "
                      "Start year and mjd are set to default values for processing:" % (receiver, base), 'blue'))
        start_yy = '21'
        start_doy = 1
        start_date = gnsscal.yrdoy2date(int('20' + start_yy), start_doy)
        start_mjd = jdcal.gcal2jd(start_date.year, start_date.month, start_date.day)[1]
        print(colored('start year %s, doy %s, mjd %s, date %s' % (start_yy, start_doy, start_mjd, start_date), 'blue'))

    return start_yy, start_mjd


""" Define RTKLIB functions """


def automate_rtklib_pp(dest_path, rover_prefix, mjd_start, mjd_end, ti_int, base_prefix, brdc_nav_prefix,
                       precise_nav_prefix, resolution, ending, options, rover_name=['NMER_original', 'NMER', 'NMLR'],
                       base_name=['LB','JB']):
    """ create input and output files for running RTKLib post processing automatically
        for all rover rinex observation files (.yyo) available in the data path directory
        (that are younger than already existing solutions)
        get doy from rover file names with name structure:
            Leica Rover: '33933650.21o' [rover + doy + '0.' + yy + 'o']
            Emlid Rover (pre-processed): 'NMER3650.21o' [rover + doy + '0.' + yy + 'o']
            Emlid Rover (original): 'ReachM2_sladina-raw_202112041058.21O' [rover + datetime + '.' + yy + 'O']
        :param dest_path:           path to GNSS rinex observation and navigation data, and rtkpost configuration file
                                    (all data needs to be in one folder)
        :param rover_prefix:        prefix of rover rinex filename
        :param mjd_start:           start modified julian date to process
        :param mjd_end:             end modified julian date to process
        :param ti_int:              processing time interval (in seconds)
        :param base_prefix:         prefix of base rinex filename
        :param brdc_nav_prefix:     prefix of broadcast navigation filename
        :param precise_nav_prefix:  prefix of precise orbit filename
        :param resolution:          processing time interval (in minutes)
        :param ending:              suffix of solution file names (e.g. a varian of processing options: '_noglonass'
        :param rover_name:          name of rover
        :param base_name:           name of base
        :param options:             rtkpost configuration file
    """
    # Q: run rtklib for all rover files in directory
    print(colored('\n\nstart processing files with RTKLIB from rover: %s and base: %s' % (rover_name, base_name), 'blue'))
    for file in glob.iglob(dest_path + rover_prefix + '*.*O', recursive=True):
        # Q: get doy from rover filenames
        rover_file = os.path.basename(file)
        if rover_name == 'NMER_original':
            # get date, year, modified julian date (mjd), doy, converted from datetime in Emlid original filename format (output from receiver, non-daily files)
            date = dt.datetime.strptime(rover_file.split('.')[0].split('_')[2], "%Y%m%d%H%M")
            year = str(date.year)[-2:]
            mjd = jdcal.gcal2jd(date.year, date.month, date.day)[1]
            doy = date.strftime('%j')
        if rover_name == 'NMER' or rover_name == 'NMLR':
            # get year, doy, date, modified julian date (mjd) directly from filename from Emlid pre-processed or Leica file name format (daily files)
            year = rover_file.split('.')[1][:2]
            doy = rover_file.split('.')[0][4:7]
            date = gnsscal.yrdoy2date(int('20' + year), int(doy))
            mjd = jdcal.gcal2jd(date.year, date.month, date.day)[1]

        # Q: only process files inbetween the selected mjd range
        if mjd_start <= mjd <= mjd_end:
            print('\nProcessing rover file: ' + rover_file, '; year: ', year, '; doy: ', doy)

            # convert doy to gpsweek and day of week (needed for precise orbit file names)
            (gpsweek, dow) = gnsscal.yrdoy2gpswd(int('20' + year), int(doy))

            # define input and output filenames (for some reason it's not working when input files are stored in subfolders!)
            base_file = base_prefix + doy + '*.' + year + 'O'
            broadcast_orbit_gps = brdc_nav_prefix + doy + '0.' + year + 'n'
            broadcast_orbit_glonass = brdc_nav_prefix + doy + '0.' + year + 'g'
            broadcast_orbit_galileo = brdc_nav_prefix + doy + '0.' + year + 'l'
            precise_orbit = precise_nav_prefix + str(gpsweek) + str(dow) + '.EPH_M'

            # create a solution directory if not existing
            sol_dir = '20_solutions/' + rover_name + '_' + base_name + '/' + resolution + '/temp_' + rover_name + '/'
            os.makedirs(dest_path + sol_dir, exist_ok=True)
            output_file = sol_dir + '20' + year + '_' + rover_name + doy + ending + '.pos'

            # Q: change directory to data directory & run RTKLib post processing command
            run_rtklib_pp(dest_path, options, ti_int, output_file, rover_file, base_file,
                          broadcast_orbit_gps, broadcast_orbit_glonass, broadcast_orbit_galileo, precise_orbit)

    print(colored('\n\nfinished processing all files with RTKLIB from rover: %s and base: %s' % (rover_name, base_name), 'blue'))


def run_rtklib_pp(dest_path, options, ti_int, output_file, rover_file, base_file, brdc_orbit_gps, brdc_orbit_glonass,
                  brdc_orbit_galileo, precise_orbit):
    """ run RTKLib post processing command (rnx2rtkp) as a subprocess (instead of manual RTKPost GUI)
        example: 'rnx2rtkp -k rtkpost_options.conf -ti 900 -o 20_solutions/NMLR/15min/NMLRdoy.pos NMLR0040.17O NMLB0040.17O NMLB0040.17n NMLB0040.17g NMLB0040.17e COD17004.eph'
        :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file (all data needs to be in one folder)
        :param options: rtkpost configuration file
        :param ti_int: processing time interval (in seconds)
        :param output_file: rtkpost solution file
        :param rover_file: GNSS observation file (rinex) from the rover receiver
        :param base_file: GNSS observation file (rinex) from the base receiver
        :param brdc_orbit_gps: GNSS broadcast (predicted) orbit for GPS satellites
        :param brdc_orbit_glonass: GNSS broadcast (predicted) orbit for GLONASS satellites
        :param brdc_orbit_galileo: GNSS broadcast (predicted) orbit for GALILEO satellites
        :param precise_orbit: GNSS precise (post processed) orbit for multi-GNSS (GPS, GLONASS, GALILEO, BEIDOU)
    """
    # change directory & run RTKLIB post processing command 'rnx2rtkp'
    process = subprocess.Popen('cd ' + dest_path + ' && rnx2rtkp '
                                                   '-k ' + options + '.conf '
                                                                     '-ti ' + ti_int + ' '
                                                                                       '-o ' + output_file + ' '
                               + rover_file + ' ' + base_file + ' ' + brdc_orbit_gps + ' ' + brdc_orbit_glonass + ' ' + brdc_orbit_galileo + ' ' + precise_orbit,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    # print(stdout) # print processing output
    print(stderr)  # print processing errors


def get_rtklib_solutions(dest_path, rover_name, resolution, ending, header_length, base_name=['LB', 'JB']):
    """  get daily rtklib ENU solution files from solution directory and store all solutions in one dataframe and pickle
    :param header_length: length of header in solution files (dependent on processing parameters)
    :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
    :param rover_name: name of rover
    :param resolution: processing time interval (in minutes)
    :param ending: suffix of solution file names (e.g. a varian of processing options: '_noglonass'
    :return: df_enu (pandas dataframe containing all seasons solution data columns ['date', 'time', 'U', 'amb_state', 'nr_sat', 'std_u'])
    """
    # Q: read all existing ENU solution data from .pkl if already exists, else create empty dataframe
    path_to_oldpickle = dest_path + '20_solutions/' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl'
    if os.path.exists(path_to_oldpickle):
        print(colored('\nReading already existing ENU solutions from pickle: %s' % path_to_oldpickle, 'yellow'))
        df_enu_old = pd.read_pickle(path_to_oldpickle)
    else:
        print(colored('\nNo existing ENU solution pickle: %s' % path_to_oldpickle, 'yellow'))
        df_enu_old = pd.DataFrame()

    # Q: read all newly available .ENU files in solution directory, parse date and time columns to datetimeindex and add them to the dataframe
    df_enu_new = pd.DataFrame(columns=['U', 'amb_state', 'nr_sat', 'std_u', 'date', 'time'])
    path = dest_path + '20_solutions/' + rover_name + '_' + base_name + '/' + resolution + '/temp_' + rover_name
    print(colored('\nReading all newly available ENU solution files from receiver: %s' % rover_name, 'blue'))
    for file in glob.iglob(path + '/*' + ending + '.pos', recursive=True):
        print('reading ENU solution file: %s' % file)
        enu = pd.read_csv(file, header=header_length, delimiter=' ', skipinitialspace=True, index_col=['date_time'],
                          na_values=["NaN"],
                          usecols=[0, 1, 4, 5, 6, 9], names=['date', 'time', 'U', 'amb_state', 'nr_sat', 'std_u'],
                          parse_dates=[['date', 'time']])

        # add new enu data to df enu
        df_enu_new = pd.concat([df_enu_new, enu], axis=0)

        # move file from temp directory to solutions directory after reading
        shutil.move(file, path + '/../' + os.path.basename(file))

    # remove date and time columns
    df_enu_new = df_enu_new.drop(columns=['date', 'time'])

    # concatenate existing solutions with new solutions
    df_enu_total = pd.concat([df_enu_old, df_enu_new], axis=0)

    # detect all dublicates and only keep last dublicated entries
    df_enu = df_enu_total[~df_enu_total.index.duplicated(keep='last')]
    print(colored('\nstored all old and new ENU solution data (without dublicates) in dataframe df_enu:', 'blue'))
    print(df_enu)

    # store dataframe as binary pickle format
    df_enu.to_pickle(dest_path + '20_solutions/' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl')
    print(colored(
        '\nstored all old and new ENU solution data (without dublicates) in pickle: '
        + '20_solutions/' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl', 'blue'))

    # delete temporary solution directory
    if os.path.exists(path):
        shutil.rmtree(path)
    print(colored('\nAll new ENU solution files are moved to solutions dir and temp solutions directory is removed',
                  'blue'))

    return df_enu


def filter_rtklib_solutions(dest_path, rover_name, base_name, baseline_length, resolution, df_enu, ambiguity=[1, 2, 5], threshold=2, window='D', ending=''):
    """ filter and clean ENU solution data (outlier filtering, median filtering, adjustments for observation mast heightening)
    :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
    :param df_enu: pandas dataframe containing all seasons solution data columns ['date', 'time', 'U (m)', 'amb_state', 'nr_sat', 'std_u (m)']
    :param rover_name: name/abbreviation of rover
    :param base_name: name/abbreviation of base
    :param baseline_length: (e.g. manually measured) distance between rover and base for calibration
    :param resolution: processing time interval (in minutes)
    :param ambiguity: ambiguity resolution state [1: fixed, 2: float, 5: standalone]
    :param threshold: set threshold for outlier removing using the standard deviation (default=2 sigma)
    :param window: window for median filter (default='D')
    :param ending: suffix of solution file names (e.g. a varian of processing options: '_noglonass'
    :return: fil_df, fil, fil_clean, m, s, jump, swe_gnss, swe_gnss_daily, std_gnss_daily
    """

    print(colored('\nFiltering data', 'blue'))

    # Q: select only data where ambiguities are fixed (amb_state==1) or float (amb_state==2) and sort datetime index
    print('\nselect data with ambiguity solution state: %s' % ambiguity)
    fil_df = pd.DataFrame(df_enu[(df_enu.amb_state == ambiguity)])
    fil_df.index = pd.DatetimeIndex(fil_df.index)
    fil_df = fil_df.sort_index()
    u = fil_df.U * 1000     # convert up (swe) component to mm

    # Q: adjust for snow mast heightening (approx. 3m elevated several times a year)
    print('\ndata is corrected for snow mast heightening events (remove sudden jumps > 1m)')

    # find positive jumps (followed by a negative jump) that would be detected as snow-mast-heightening but are outliers
    jump_or_outlier = u[(u.diff() > 1000)]
    for i, j in enumerate(jump_or_outlier):
        # grap the next row after detected jump
        next_line = u[u.index > jump_or_outlier.index[i]].head(1)
        # check if this is really an outlier (data switches back to values from before) or a data jump (stays at similar values)
        if next_line[0] < jump_or_outlier[i] - 1000:
            # delete if outlier
            u = u.drop(jump_or_outlier.index[i])

    # find negative jumps (followed by a positive jump)
    value_after_jump = u[(u.diff() < -1000)]
    for i, j in enumerate(value_after_jump):
        next_line = u[u.index > value_after_jump.index[i]].head(1)
        if next_line[0] > value_after_jump[i]+1000:
            u = u.drop(value_after_jump.index[i])

    # After erasing outliers/false SMH-jumps, find true SMH-jumps
    jump = u.diff()[(u.diff() < -1000)]

    # get value of jump difference (of values directly after - before jump)
    jump_ind = jump.index.format()[0]
    jump_val = u[jump_ind] - u[:jump_ind][-2]

    while jump.empty is False:
        print('\njump of %s mm height is detected! at %s' % (jump_val, jump.index.format()[0]))
        adj = u[(u.index >= jump.index.format()[0])] - jump_val  # correct all observations after jump [0]
        u = pd.concat([u[~(u.index >= jump.index.format()[0])],
                       adj])  # concatenate all original obs before jump with adjusted values after jump
        jump = u.diff()[(u.diff() < -1000)]

    print('\nno jump detected!')

    # Q: remove outliers based on x*sigma threshold
    print('\nremove outliers based on %s * sigma threshold' % threshold)
    upper_limit = u.rolling('3D').median() + threshold * u.rolling('3D').std()
    lower_limit = u.rolling('3D').median() - threshold * u.rolling('3D').std()
    u_clean = u[(u > lower_limit) & (u < upper_limit)]

    # Q: correct values to be positive values by subtracting length of true baseline
    swe_gnss = u_clean - baseline_length
    swe_gnss.index = swe_gnss.index + pd.Timedelta(seconds=18)
    date_of_min = swe_gnss[swe_gnss == swe_gnss.min()].index

    # Q: filter data with a rolling median
    print('\ndata is median filtered with window length: %s' % window)
    swe_gnss_fil = swe_gnss.rolling(window).median()
    std_gnss_fil = swe_gnss.rolling(window).std()

    # resample data per day, calculate median and standard deviation (noise) per day to fit manual reference data
    swe_gnss_daily = swe_gnss.resample('D').median()
    std_gnss_daily = swe_gnss.resample('D').std()
    std_gnss_mean = std_gnss_daily.mean()
    std_gnss_percentual = std_gnss_daily * 100 / swe_gnss_daily
    std_gnss_percentual_mean = std_gnss_percentual.mean()
    print('\nThe daily noise of %s - %s solution (mean of standard deviation of 15min solution from daily mean):'
          '%s kg/m2 (%s %%)' % (base_name, rover_name, std_gnss_mean, std_gnss_percentual_mean))

    # Q: store swe results to pickle
    print(colored(
        '\ndata is filtered, cleaned, and corrected and SWE results are stored to pickle and .csv: %s' % '20_solutions/SWE_results/swe_gnss_' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl',
        'blue'))
    os.makedirs(dest_path + '20_solutions/SWE_results/', exist_ok=True)
    swe_gnss.to_pickle(
        dest_path + '20_solutions/SWE_results/swe_gnss_' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl')
    swe_gnss.to_csv(dest_path + '20_solutions/SWE_results/swe_gnss_' + rover_name + '_' + base_name + '_' + resolution + ending + '.csv')

    return fil_df, u, u_clean, swe_gnss, swe_gnss_fil, std_gnss_fil, swe_gnss_daily, std_gnss_daily, std_gnss_percentual, date_of_min


def read_swe_gnss(dest_path, swe_gnss, rover_name, resolution, ending, base_name):
    # read gnss swe results from pickle
    if swe_gnss is None:
        print(colored(
            '\nSWE results are NOT available, reading from pickle: %s' % '20_solutions/SWE_results/swe_gnss_' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl',
            'orange'))
        swe_gnss = pd.read_pickle(
            dest_path + '20_solutions/SWE_results/swe_gnss_' + rover_name + '_' + base_name + '_' + resolution + ending + '.pkl')

    return swe_gnss


""" Define reference sensors functions """


def read_manual_observations(dest_path):
    """ read and interpolate manual accumulation (cm), density (kg/m^3), SWE (mm w.e.) data
        :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
        :return: manual2, ipol
    """
    # create local directory for reference observations
    loc_ref_dir = dest_path + '00_reference_data/'
    os.makedirs(loc_ref_dir, exist_ok=True)

    # read data
    print('\nread manual observations')
    manual_NEW = pd.read_csv(loc_ref_dir + 'snowpit_density.csv', header=1, skipinitialspace=True,
                             delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True,
                             names=['GNSS-IR snowhight (16.11. sh = 0) [mm]',
                                    'density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]',
                                    'depth snow pit [mm]', 'density / snow pit profile [kg/m³]',
                                    'new snow since last measurement [cm]',
                                    'density/ new snow layer [kg/m³]'])

    manual = pd.read_csv(loc_ref_dir + 'Manual_Spuso.csv', header=1, skipinitialspace=True,
                         delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True,
                         names=['Acc', 'Density', 'SWE', 'Density_aboveAnt', 'SWE_aboveAnt'])
    manual2 = manual
    manual2.index = manual2.index + pd.Timedelta(days=0.2)

    # fix dtype of column "Acc" and convert to mm
    manual2.Acc = manual2.Acc.astype('float64') * 10


    # interpolate manual data
    print('\n-- interpolate manual reference observations to minute intervals')
    ipol = manual_NEW['density / GNSS-IR sh (snow density of snowpack above antenna) [kg/m³]'].resample('min').interpolate(method='linear', limit_direction='backward')

    return manual2, manual_NEW, ipol


def read_snowbuoy_observations(dest_path, url, ipol_density=None):
    """ read snow buoy accumulation data from four sensors and convert to SWE & pressure, airtemp
        :param ipol_density: interpolated density data from manual reference observations
        :param url: webpage url where daily updated snow buoy data  can be downloaded
        :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
        :return: buoy
    """
    # create local directory for snow buoy observations
    loc_buoy_dir = dest_path + '00_reference_data/Snowbuoy/'
    os.makedirs(loc_buoy_dir, exist_ok=True)

    # Q: download newest snow buoy data from url
    # get data from url
    r = requests.get(url, allow_redirects=True)

    # decompress file
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # store selected file from decompressed folder to working direcory subfolder
    z.extract(z.filelist[2], path=loc_buoy_dir)

    # Q: read snow buoy data
    print('\nread snow buoy observations')
    buoy_all = pd.read_csv(loc_buoy_dir + '2017S54_300234011695900_proc.csv', header=0,
                           skipinitialspace=True, delimiter=',', index_col=0, skiprows=0, na_values=["NaN"],
                           parse_dates=[0],
                           names=['lat', 'lon', 'sh1', 'sh2', 'sh3', 'sh4', 'pressure', 'airtemp', 'bodytemp',
                                  'gpstime'])

    # select only accumulation data from season 21/22 & convert to mm
    buoy = buoy_all[['sh1', 'sh2', 'sh3', 'sh4']]['2021-11-26':] * 1000

    # Q: adjust for snow mast heightening (approx. 1m elevated); value of jump difference (of values directly after - before jump): 2023-01-24 21:01:00 1036.0
    print('\ndata is corrected for snow buoy heightening events (remove sudden jumps > 1m)')
    jump_ind = '2023-01-24 21:01:00'
    jump_val = 1036

    print('\ncorrect jump of height %s: at %s' % (jump_val, jump_ind))
    adj = buoy[['sh1', 'sh2', 'sh3', 'sh4']][
              (buoy.index >= jump_ind)] + jump_val  # correct all observations after jump [0]
    buoy_corr = pd.concat([buoy[['sh1', 'sh2', 'sh3', 'sh4']][~(buoy.index >= jump_ind)],
                           adj])  # concatenate all original obs before jump with adjusted values after jump

    # Q: Differences in accumulation & conversion to SWE
    # calculate change in accumulation (in mm) for each buoy sensor add it as an additional column to the dataframe buoy
    buoy_change = buoy_corr[['sh1', 'sh2', 'sh3', 'sh4']] - buoy_corr[['sh1', 'sh2', 'sh3', 'sh4']].iloc[0]
    buoy_change.columns = ['dsh1', 'dsh2', 'dsh3', 'dsh4']

    # convert snow accumulation to SWE (with interpolated and constant density values)
    print('\n-- convert buoy observations to SWE')
    buoy_swe = convert_sh2swe(buoy_change, ipol_density)
    buoy_swe.columns = ['dswe1', 'dswe2', 'dswe3', 'dswe4']

    buoy_swe_constant = convert_sh2swe(buoy_change)
    buoy_swe_constant.columns = ['dswe_const1', 'dswe_const2', 'dswe_const3', 'dswe_const4']

    # append new columns to existing buoy dataframe
    buoy = pd.concat([buoy_corr, buoy_change, buoy_swe, buoy_swe_constant], axis=1)

    return buoy


def read_pole_observations(dest_path, ipol_density=None):
    """ read Pegelfeld Spuso accumulation data from 16 poles and convert to SWE
        :param ipol_density: interpolated density data from manual reference observations
        :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
        :return: poles
    """
    # create local directory for reference observations
    loc_ref_dir = dest_path + '00_reference_data/'
    os.makedirs(loc_ref_dir, exist_ok=True)

    # Q: read Pegelfeld Spuso pole observations
    print('\nread Pegelfeld Spuso pole observations')
    poles = pd.read_csv(loc_ref_dir + 'Pegelfeld_Spuso_Akkumulation.csv', header=0, delimiter=';',
                        index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True)

    # Q: convert snow accumulation to SWE (with interpolated and constant density values)
    print('\n-- convert Pegelfeld Spuso pole observations to SWE')
    poles_swe = convert_sh2swe(poles, ipol_density)
    poles_swe.columns = ['dswe'] + poles_swe.columns

    poles_swe_constant = convert_sh2swe(poles)
    poles_swe_constant.columns = ['dswe_const'] + poles_swe_constant.columns

    # append new columns to existing poles dataframe
    poles = pd.concat([poles, poles_swe, poles_swe_constant], axis=1)

    return poles

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def read_mob_data(dest_path, mob_path, yy, mob_pickle='nm_mob'):
    # create local directory for meteorlogic observations
    loc_mob_dir = dest_path + '00_reference_data/mob/'
    os.makedirs(loc_mob_dir, exist_ok=True)

    # Q: copy laser observations (*.val) from AWI server if not already existing
    print(colored("\ncopy new meteorologic observation files", 'blue'))
    # get list of yearly directories newer than first year
    for year in os.listdir(mob_path):
        if has_numbers(year) is True and int(year) >= int('20' + yy):
            # copy missing laser observation files
            for f in glob.glob(mob_path + year + '/nm*.val'):
                file = os.path.basename(f)
                # skip files of 2021 before 26th nov (no gps data before installation)
                if int(file[2:8]) >= 211101:
                    if not os.path.exists(loc_mob_dir + file):
                        shutil.copy2(f, loc_mob_dir)
                        print("file copied from %s to %s" % (f, loc_mob_dir))
                    else:
                        pass
                else:
                    pass
        else:
            pass
    print(colored("\nnew meteorologic observation (mob) files copied", 'blue'))

    # Q: read all existing laser observations from .pkl if already exists, else create empty dataframe
    path_to_oldpickle = loc_mob_dir + mob_pickle + '.pkl'
    if os.path.exists(path_to_oldpickle):
        print('\nReading already existing mob observations from pickle: %s' % path_to_oldpickle)
        mob = pd.read_pickle(path_to_oldpickle)
        old_idx = mob.index[-1].date().strftime("%y%m%d")
    else:
        print(colored('\nCreating a mob observations pickle', 'yellow'))
        mob = pd.DataFrame()
        old_idx = '211101'

    # Q: read new files *.val, parse date and time columns to datetimeindex and add them to the dataframe
    print(colored('\nReading all new logfiles from: %s' % loc_mob_dir + 'nm*.val', 'blue'))
    for file in glob.iglob(loc_mob_dir + 'nm*.val', recursive=True):
        # read files newer than last entry in mob pickle
        if int(os.path.basename(file)[2:8]) > int(old_idx):
            print(file)

            # check if type of mob data format to read - new file header
            if int(os.path.basename(file)[2:8]) < 230105:
                # header NamesLong	2021-03-09 00:00:00	2023-01-05 00:00:00	"Hour:Minute	Cloud Ceiling	Visibility	Sunshine Indicator	Shortwave Downward Radiant Energy Flux Density	Shortwave Upward Radiant Energy Flux Density	RG8 Filtered Downward Radiant Energy Flux Density	OG1 Filtered Downward Radiant Energy Flux Density	Diffuse Shortwave Downward Radiant Energy Flux Density	Direct Shortwave Downward Radiant Energy Flux Density	UV Filtered Downward Radiant Energy Flux Density	Longwave Downward Radiant Energy Flux Density	Longwave Downward Radiant Energy Flux Density CGR4	Longwave Upward Radiant Energy Flux Density	Longwave Downward Body Temperature	Cgr4 Longwave Upward Body Temperature	Longwave Upward Body Temperature	2m Level Wind Speed	2m Level Wind Direction	10m Level Wind Speed	2m Level Wind Direction	10m Level Gust Wind Speed	2m Level Temperature	10m Level Temperature	Relative Humidity HMT337 in T2 housing	Relative Humidity HMT337 in T10 housing	Relative Humidity HMP155T2	Relative Humidity HMP155T10	Air Pressure	Air Pressure	Air Pressure	Longwave Downward Dome Temperature 1	Longwave Downward Dome Temperature 2	Longwave Downward Dome Temperature 3	Longwave Upward Dome Temperature 1	Longwave Upward Dome Temperature 2	Longwave Upward Dome Temperature 3	not in use	not in use	not in use	Direct Shortwave Downward Body Temperature	Shortwave Downward Body Temperature	Shortwave Upward Body Temperature	Snow Level Height	Snow Level Sensor Signal Strength"
                mob_new = pd.read_csv(file, header=0, na_values=[-999.9], delim_whitespace=True,
                                      names=['datetime', 'Cloud Ceiling', 'Visibility', 'Sunshine Indicator', 'Shortwave Downward Radiant Energy Flux Density', 'Shortwave Upward Radiant Energy Flux Density', 'RG8 Filtered Downward Radiant Energy Flux Density', 'OG1 Filtered Downward Radiant Energy Flux Density', 'Diffuse Shortwave Downward Radiant Energy Flux Density', 'Direct Shortwave Downward Radiant Energy Flux Density', 'UV Filtered Downward Radiant Energy Flux Density', 'Longwave Downward Radiant Energy Flux Density1', 'Longwave Downward Radiant Energy Flux Density2', 'Longwave Upward Radiant Energy Flux Density', 'Longwave Downward Body Temperature', 'Cgr4 Longwave Upward Body Temperature', 'Longwave Upward Body Temperature', '2m Level Wind Speed', '2m Level Wind Direction', '10m Level Wind Speed', '10m Level Wind Direction', '10m Level Gust Wind Speed', '2m Level Temperature', '10m Level Temperature', 'Relative Humidity HMT337 in T2 housing', 'Relative Humidity HMT337 in T10 housing', 'Relative Humidity HMP155T2', 'Relative Humidity HMP155T10', 'Air Pressure 1', 'Air Pressure 2', 'Air Pressure 3', 'Longwave Downward Dome Temperature 1', 'Longwave Downward Dome Temperature 2', 'Longwave Downward Dome Temperature 3', 'Longwave Upward Dome Temperature 1', 'Longwave Upward Dome Temperature 2', 'Longwave Upward Dome Temperature 3', 'not in use 1', 'not in use 2', 'not in use 3', 'Direct Shortwave Downward Body Temperature', 'Shortwave Downward Body Temperature', 'Shortwave Upward Body Temperature', 'Snow Level Height', 'Snow Level Sensor Signal Strength'],
                                      usecols=[0, 3, 17, 18, 19, 20, 21, 22, 23, 43],
                                      encoding='latin1', engine='python', dayfirst=True)
            elif int(os.path.basename(file)[2:8]) >= 230105 and int(os.path.basename(file)[2:8]) <	230218:
                # header NamesLong	2023-01-05 00:00:00	2023-02-18 00:00:00	"Hour:Minute	Cloud Ceiling	Visibility	Sunshine Indicator	Shortwave Downward Radiant Energy Flux Density	Shortwave Upward Radiant Energy Flux Density	RG8 Filtered Downward Radiant Energy Flux Density	OG1 Filtered Downward Radiant Energy Flux Density	Diffuse Shortwave Downward Radiant Energy Flux Density	Direct Shortwave Downward Radiant Energy Flux Density	UV Filtered Downward Radiant Energy Flux Density	Longwave Downward Radiant Energy Flux Density	Longwave Downward Radiant Energy Flux Density CGR4	Longwave Upward Radiant Energy Flux Density	Longwave Downward Body Temperature	Cgr4 Longwave Upward Body Temperature	Longwave Upward Body Temperature	2m Level Wind Speed	2m Level Wind Direction	10m Level Wind Speed	2m Level Wind Direction	10m Level Gust Wind Speed	2m Level Temperature	10m Level Temperature	Relative Humidity HMP155T2	Relative Humidity HMP155T10	Air Pressure	Air Pressure	Air Pressure	Longwave Downward Dome Temperature 1	Longwave Downward Dome Temperature 2	Longwave Downward Dome Temperature 3	Longwave Upward Dome Temperature 1	Longwave Upward Dome Temperature 2	Longwave Upward Dome Temperature 3	not in use	not in use	not in use	Direct Shortwave Downward Body Temperature	Shortwave Downward Body Temperature	Shortwave Upward Body Temperature	Snow Level Height	Snow Level Sensor Signal Strength"
                mob_new = pd.read_csv(file, header=0, na_values=[-999.9], delim_whitespace=True,
                                      names=['datetime', 'Cloud Ceiling', 'Visibility', 'Sunshine Indicator', 'Shortwave Downward Radiant Energy Flux Density', 'Shortwave Upward Radiant Energy Flux Density', 'RG8 Filtered Downward Radiant Energy Flux Density', 'OG1 Filtered Downward Radiant Energy Flux Density', 'Diffuse Shortwave Downward Radiant Energy Flux Density', 'Direct Shortwave Downward Radiant Energy Flux Density', 'UV Filtered Downward Radiant Energy Flux Density', 'Longwave Downward Radiant Energy Flux Density1', 'Longwave Downward Radiant Energy Flux Density2', 'Longwave Upward Radiant Energy Flux Density', 'Longwave Downward Body Temperature', 'Cgr4 Longwave Upward Body Temperature', 'Longwave Upward Body Temperature', '2m Level Wind Speed', '2m Level Wind Direction', '10m Level Wind Speed', '10m Level Wind Direction', '10m Level Gust Wind Speed', '2m Level Temperature', '10m Level Temperature', 'Relative Humidity HMP155T2', 'Relative Humidity HMP155T10', 'Air Pressure 1', 'Air Pressure 2', 'Air Pressure 3', 'Longwave Downward Dome Temperature 1', 'Longwave Downward Dome Temperature 2', 'Longwave Downward Dome Temperature 3', 'Longwave Upward Dome Temperature 1', 'Longwave Upward Dome Temperature 2', 'Longwave Upward Dome Temperature 3', 'not in use 1', 'not in use 2', 'not in use 3', 'Direct Shortwave Downward Body Temperature', 'Shortwave Downward Body Temperature', 'Shortwave Upward Body Temperature', 'Snow Level Height', 'Snow Level Sensor Signal Strength'],
                                      usecols=[0, 3, 17, 18, 19, 20, 21, 22, 23, 41],
                                      encoding='latin1', engine='python')
            elif int(os.path.basename(file)[2:8]) >= 230218:
                # header NamesLong	2023-02-18 00:00:00		                "Hour:Minute	Cloud Ceiling	Visibility	Sunshine Indicator	Shortwave Downward Radiant Energy Flux Density	Shortwave Upward Radiant Energy Flux Density	RG8 Filtered Downward Radiant Energy Flux Density	OG1 Filtered Downward Radiant Energy Flux Density	Diffuse Shortwave Downward Radiant Energy Flux Density	Direct Shortwave Downward Radiant Energy Flux Density	UV Filtered Downward Radiant Energy Flux Density	Longwave Downward Radiant Energy Flux Density	Longwave Downward Radiant Energy Flux Density CGR4	Longwave Upward Radiant Energy Flux Density	Longwave Downward Body Temperature	Cgr4 Longwave Upward Body Temperature	Longwave Upward Body Temperature	2m Level Wind Speed	2m Level Wind Direction	10m Level Wind Speed	2m Level Wind Direction	10m Level Gust Wind Speed	2m Level Temperature	10m Level Temperature	Relative Humidity HMP155T2	Relative Humidity HMP155T10	Air Pressure	Air Pressure	Air Pressure	Longwave Downward Dome Temperature 1	Longwave Downward Dome Temperature 2	Longwave Downward Dome Temperature 3	Longwave Upward Dome Temperature 1	Longwave Upward Dome Temperature 2	Longwave Upward Dome Temperature 3	not in use	not in use	not in use	Direct Shortwave Downward Body Temperature	Ventilation Temperature Mast	Ventilation Temperature GLC	Snow Level Height	Snow Level Sensor Signal Strength"
                mob_new = pd.read_csv(file, header=0, na_values=[-999.9], delim_whitespace=True,
                                      names=['datetime', 'Cloud Ceiling', 'Visibility', 'Sunshine Indicator', 'Shortwave Downward Radiant Energy Flux Density', 'Shortwave Upward Radiant Energy Flux Density', 'RG8 Filtered Downward Radiant Energy Flux Density', 'OG1 Filtered Downward Radiant Energy Flux Density', 'Diffuse Shortwave Downward Radiant Energy Flux Density', 'Direct Shortwave Downward Radiant Energy Flux Density', 'UV Filtered Downward Radiant Energy Flux Density', 'Longwave Downward Radiant Energy Flux Density1', 'Longwave Downward Radiant Energy Flux Density2', 'Longwave Upward Radiant Energy Flux Density', 'Longwave Downward Body Temperature', 'Cgr4 Longwave Upward Body Temperature', 'Longwave Upward Body Temperature', '2m Level Wind Speed', '2m Level Wind Direction', '10m Level Wind Speed', '10m Level Wind Direction', '10m Level Gust Wind Speed', '2m Level Temperature', '10m Level Temperature', 'Relative Humidity HMP155T2', 'Relative Humidity HMP155T10', 'Air Pressure 1', 'Air Pressure 2', 'Air Pressure 3', 'Longwave Downward Dome Temperature 1', 'Longwave Downward Dome Temperature 2', 'Longwave Downward Dome Temperature 3', 'Longwave Upward Dome Temperature 1', 'Longwave Upward Dome Temperature 2', 'Longwave Upward Dome Temperature 3', 'not in use 1', 'not in use 2', 'not in use 3', 'Direct Shortwave Downward Body Temperature', 'Ventilation Temperature Mast', 'Ventilation Temperature GLC', 'Snow Level Height', 'Snow Level Sensor Signal Strength'],
                                      usecols=[0, 3, 17, 18, 19, 20, 21, 22, 23, 41],
                                      encoding='latin1', engine='python')

            # create datetime index
            mob_new.datetime = os.path.basename(file)[2:8] + mob_new.datetime
            mob_new.datetime = pd.to_datetime(mob_new['datetime'], format='%y%m%d%H:%M')
            mob_new = mob_new.set_index('datetime')
            # add loaded file to existing mob df
            mob = pd.concat([mob, mob_new], axis=0)

        else:
            continue

    # detect all dublicates and only keep last dublicated entries
    mob = mob[~mob.index.duplicated(keep='last')]

    # store as .pkl
    mob.to_pickle(loc_mob_dir + mob_pickle + '.pkl')
    print('\nstored all old and new mob observations (without dublicates) to pickle: %s' + loc_mob_dir + mob_pickle + '.pkl')

    return mob


def read_SYNOP_data(dest_path, synop_path, yy, pickle='nm_SYNOP'):
    # create local directory for meteorlogic observations
    loc_synop_dir = dest_path + '00_reference_data/SYNOP/'
    os.makedirs(loc_synop_dir, exist_ok=True)

    # Q: copy laser observations (*.val) from AWI server if not already existing
    print(colored("\ncopy new synoptic observation files", 'blue'))
    # get list of yearly directories newer than first year
    for year in os.listdir(synop_path):
        if has_numbers(year) is True and int(year) >= int('20' + yy):
            # copy missing laser observation files
            for f in glob.glob(synop_path + year + '/nm*.archive'):
                file = os.path.basename(f)
                # skip files of 2021 before 26th nov (no gps data before installation)
                if int(file[2:6]) >= 2111:
                    if not os.path.exists(loc_synop_dir + file):
                        shutil.copy2(f, loc_synop_dir)
                        print("file copied from %s to %s" % (f, loc_synop_dir))
                    else:
                        pass
                else:
                    pass
        else:
            pass
    print(colored("\nnew SYNOP observation files copied", 'blue'))

    # Q: read all existing laser observations from .pkl if already exists, else create empty dataframe
    path_to_oldpickle = loc_synop_dir + pickle + '.pkl'
    if os.path.exists(path_to_oldpickle):
        print('\nReading already existing synoptic observations from pickle: %s' % path_to_oldpickle)
        synop = pd.read_pickle(path_to_oldpickle)
        old_idx = synop.index[-1].date().strftime("%y%m")
    else:
        print(colored('\nCreating a SYNOP observations pickle', 'yellow'))
        synop = pd.DataFrame()
        old_idx = '2110'

    # Q: read new files *.archive, parse date and time columns to datetimeindex and add them to the dataframe
    print(colored('\nReading all new logfiles from: %s' % loc_synop_dir + 'nm*.archive', 'blue'))
    for file in glob.iglob(loc_synop_dir + 'nm*.archive', recursive=True):
        # read files newer than last entry in mob pickle
        if int(os.path.basename(file)[2:6]) >= int(old_idx):
            print(file)

            # only save present weather (ww) condition in dataframe
            synop_new = pd.read_csv(file, header=0, na_values=['7////'], delim_whitespace=True,
                                  names=['date', 'AAXX', 'time', 'station number', '42999', '42209', 'T', '21075', '39730', '49786', '57004', 'ww', '83031', 'new set', '10006', '21061', '929//', '/////'],
                                  usecols=[0, 2, 11], parse_dates=[['date', 'time']],
                                  encoding='latin1', engine='python', dayfirst=True)

            if synop_new.empty is True:
                for f in glob.glob(loc_synop_dir + os.listdir(loc_synop_dir)[-2]):
                    os.remove(f)
            else:
                # create datetime index
                synop_new['date_time'] = synop_new['date_time'].astype(str).str[:-1]
                synop_new['date_time'] = pd.to_datetime(synop_new['date_time'], format='%y%m %d%H')
                synop_new = synop_new.set_index('date_time')
                # take only numbers that represent weather
                synop_new['ww'] = synop_new['ww'].astype(str).str[1:-4]
                synop_new['ww'] = pd.to_numeric(synop_new.ww)
                # add loaded file to existing mob df
                synop = pd.concat([synop, synop_new], axis=0)

        else:
            continue

    # detect all dublicates and only keep last dublicated entries
    synop = synop[~synop.index.duplicated(keep='last')]

    # store as .pkl
    synop.to_pickle(loc_synop_dir + pickle + '.pkl')
    print('\nstored all old and new mob observations (without dublicates) to pickle: %s' + loc_synop_dir + pickle + '.pkl')

    return synop



def read_laser_observations(dest_path, laser_path, yy, laser_pickle='nm_laser'):
    """ read snow accumulation observations (minute resolution) from laser distance sensor data
    :param ipol: interpolated density data from manual reference observations
    :param laser_pickle: read laser pickle (e.g., '00_reference_data/Laser/nm_shm.pkl') and logfiles creating/containing snow accumulation observations from laser distance sensor
    :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
    :return: df_shm, h, fil_h_clean, h_resampled, h_std_resampled, sh, sh_std
    """
    # create local directory for laser observations
    loc_laser_dir = dest_path + '00_reference_data/Laser/'
    os.makedirs(loc_laser_dir, exist_ok=True)

    # Q: copy laser observations (*.log/shm = *.[ls]??) from AWI server if not already existing
    print(colored("\ncopy new laser files", 'blue'))
    # get list of yearly directories newer than first year
    for year in os.listdir(laser_path)[:-1]:
        if int(year) >= int('20' + yy):
            # copy missing laser observation files
            for f in glob.glob(laser_path + year + '/*.[ls]??'):
                file = os.path.basename(f)
                # skip files of 2021 before 26th nov (no gps data before installation)
                if int(file[2:8]) >= 211101:
                    if not os.path.exists(loc_laser_dir + file):
                        shutil.copy2(f, loc_laser_dir)
                        print("file copied from %s to %s" % (f, loc_laser_dir))
                    else:
                        # print(colored("\nfile in destination already exists: %s, \ncopy aborted!!!" % dest_path, 'yellow'))
                        pass
                else:
                    pass
        else:
            pass
    print(colored("\nnew laser files copied", 'blue'))

    # Q: read all existing laser observations from .pkl if already exists, else create empty dataframe
    path_to_oldpickle = loc_laser_dir + laser_pickle + '.pkl'
    if os.path.exists(path_to_oldpickle):
        print(colored('\nReading already existing laser observations from pickle: %s' % path_to_oldpickle, 'yellow'))
        laser = pd.read_pickle(path_to_oldpickle)
        old_idx = laser.index[-1].date().strftime("%y%m%d")
    else:
        print(colored('\nNo existing laser observations pickle!', 'yellow'))
        laser = pd.DataFrame()
        old_idx = '211101'

    # Q: read new snow accumulation files *.[log/shm] (minute resolution) from laser distance sensor data, parse date and time columns to datetimeindex and add them to the dataframe
    print(colored('\nReading all new logfiles from: %s' % loc_laser_dir + 'nm*.[log/shm]', 'blue'))
    for file in glob.iglob(loc_laser_dir + 'nm*.[ls]??', recursive=True):
        # read accumulation files newer than last entry in laser pickle
        if int(os.path.basename(file)[2:8]) > int(old_idx):
            print(file)

            # check if old or new type laser data format to read due to the installation of a new sensor on 22-12-2022
            if int(os.path.basename(file)[2:8]) <= 221222:
                # read all old-type snow accumulation.log files
                # header: 'date', 'time', 'snow level (m)', 'signal(-)', 'temp (°C)', 'error (-)', 'checksum (-)'
                shm = pd.read_csv(file, header=0, delimiter=r'[ >]', skipinitialspace=True, na_values=["NaN"],
                                  names=['date', 'time', 'none', 'sh', 'signal', 'temp', 'error', 'check'],
                                  usecols=[0, 1, 3, 5, 6],
                                  encoding='latin1', parse_dates=[['date', 'time']], index_col=['date_time'],
                                  engine='python', dayfirst=True)
            else:
                # read all new-type snow accumulation.shm files
                # header: Year	Month	Day	Hour	Minute	Second	Command	TelegramNumber	SerialNumber	SnowLevel	SnowSignal	Temperature	TiltAngle	Error	UmbStatus	Checksum	DistanceRaw	Unknown	Checksum660
                shm = pd.read_csv(file, header=0, delimiter=' |;', na_values=["NaN"],
                                  names=['datetime', 'Command', 'TelegramNumber', 'SerialNumber', 'sh', 'signal',
                                         'temp', 'TiltAngle', 'error', 'check'],
                                  usecols=[0, 4, 6, 8],
                                  encoding='latin1', parse_dates=['datetime'], index_col=0,
                                  engine='python')
                # only select error infos in 'error' (first column)
                shm.error = shm.error.str.split(':', expand=True)[0]
                # change outlier values ('///////') to NaN
                shm.sh = pd.to_numeric(shm.sh, errors='coerce')
                shm.error = pd.to_numeric(shm.error, errors='coerce')

            # add loaded file to existing laser df
            laser = pd.concat([laser, shm], axis=0)

        else:
            continue

    # calculate change in accumulation (in mm) and add it as an additional column to the dataframe
    laser['dsh'] = (laser['sh'] - laser['sh'][0]) * 1000

    # detect all dublicates and only keep last dublicated entries
    laser = laser[~laser.index.duplicated(keep='last')]

    # store as .pkl
    laser.to_pickle(loc_laser_dir + laser_pickle + '.pkl')
    print(colored(
        '\nstored all old and new laser observations (without dublicates) to pickle: %s' + loc_laser_dir + laser_pickle + '.pkl',
        'blue'))

    return laser


def filter_laser_observations(ipol, laser, threshold=1):
    """ filter snow accumulation observations (minute resolution) from laser distance sensor data
    :param threshold: threshold for removing outliers (default=1)
    :param ipol: interpolated density data from manual reference observations
    :param laser: laser data
    :return: laser_filtered
    """

    # Q: remove outliers in laser observations
    print('\n-- filtering laser observations')
    # 0. select only observations without errors
    dsh = laser[(laser.error == 0)].dsh

    # 1. remove huge outliers
    f = dsh[(dsh > dsh.min())]

    # 2. remove outliers based on an x sigma threshold
    print('\nremove outliers based on %s * sigma threshold' % threshold)
    upper_limit = f.rolling('7D').median() + threshold * f.rolling('7D').std()
    lower_limit = f.rolling('7D').median() - threshold * f.rolling('7D').std()
    f_clean = f[(f > lower_limit) & (f < upper_limit)]

    # 3. remove remaining outliers based on their gradient
    print('\nremove outliers based on gradient')
    gradient = f_clean.diff()
    outliers = f_clean.index[(gradient > 500) | (gradient < -500)]
    while outliers.empty is False:
        fil_dsh = f_clean.loc[~f_clean.index.isin(outliers)]
        f_clean = fil_dsh
        gradient = f_clean.diff()
        outliers = f_clean.index[(gradient > 500) | (gradient < -500)]

    # Q: filter observations
    print('\nmedian filtering')
    laser_fil = f_clean.rolling('D').median()
    laser_fil_std = f_clean.rolling('D').std()

    # Q: calculate SWE from accumulation data
    print('\n-- convert laser observations to SWE')
    laser_swe = convert_sh2swe(laser_fil, ipol_density=ipol)
    laser_swe_constant = convert_sh2swe(laser_fil)

    # append new columns to existing laser dataframe
    laser_filtered = pd.concat([laser_fil, laser_fil_std, laser_swe, laser_swe_constant], axis=1)
    laser_filtered.columns = ['dsh', 'dsh_std', 'dswe', 'dswe_const']

    return laser_filtered


def read_reference_data(dest_path, laser_path, mob_path, synop_path, yy, url, read_manual=[True, False], read_buoy=[True, False],
                        read_poles=[True, False], read_laser=[True, False], read_mob=[True, False], read_synop=[True, False],
                        laser_pickle='00_reference_data/Laser/nm_laser.pkl',
                        mob_pickle='00_reference_data/mob/nm_mob.pkl',
                        synop_pickle='00_reference_data/SYNOP/nm_synop.pkl'):
    """ read reference sensor's observations from manual observations, a snow buoy sensor, a laser distance sensor and manual pole observations
    :param read_laser: read laser accumulation data (True) or not (False)
    :param read_poles: read poles accumulation data (True) or not (False)
    :param read_buoy: read buoy accumulation data (True) or not (False)
    :param read_manual: read manual observation data (True) or not (False)
    :param laser_pickle: read logfiles (laser_pickle == None) or pickle (e.g., '00_reference_data/Laser/nm_laser.pkl') creating/containing snow accumulation observations from laser distance sensor
    :param dest_path: path to GNSS rinex observation and navigation data, and rtkpost configuration file
    :param url: path to snow buoy data on a webpage
    :param yy: year to get first data
    :param laser_path: path to laser distance sensor observation files
    :return: manual, ipol, buoy, poles, laser, laser_filtered
    """
    print(colored('\n\nread reference observations', 'blue'))

    # Q: read manual accumulation (cm), density (kg/m^3), SWE (mm w.e.) data
    if read_manual is True:
        manual, manual_NEW, ipol = read_manual_observations(dest_path)
        manual.index = pd.DatetimeIndex(manual.index.date)
        manual_NEW.index = pd.DatetimeIndex(manual_NEW.index.date)
    else:
        manual, manual_NEW, ipol = None, None, None

    # Q: read snow buoy data (mm)
    if read_buoy is True:
        buoy = read_snowbuoy_observations(dest_path, url, ipol_density=ipol)
    else:
        buoy = None

    # Q: read Pegelfeld Spuso accumulation data from poles
    if read_poles is True:
        poles = read_pole_observations(dest_path, ipol_density=ipol)
    else:
        poles = None

    # Q: read snow depth observations (minute resolution) from laser distance sensor data
    if read_laser is True:
        laser = read_laser_observations(dest_path, laser_path, yy, laser_pickle)
        laser_filtered = filter_laser_observations(ipol, laser, threshold=1)
    else:
        laser, laser_filtered = None, None

    if read_mob is True:
        mob = read_mob_data(dest_path, mob_path, yy, mob_pickle)
    else:
        mob = None

    if read_synop is True:
        synop = read_SYNOP_data(dest_path, synop_path, yy, synop_pickle)
    else:
        synop = None

    print(colored('\n\nreference observations are loaded', 'blue'))

    if read_laser is True and read_manual is False and read_buoy is False and read_poles is False and read_mob is False and read_synop is False:
        return laser, laser_filtered
    elif read_laser is False and read_manual is True and read_buoy is True and read_poles is True and read_mob is True and read_synop is True:
        return manual, manual_NEW, ipol, buoy, poles, mob, synop
    else:
        return manual, manual_NEW, ipol, buoy, poles, laser, laser_filtered, mob, synop


def calc_footprint(incident_angle):
    snowheight = np.linspace(0, 2, 201)
    footprint_drysnow = func_footprint(snowheight, 1.32, incident_angle)
    footprint_moistsnow = func_footprint(snowheight, 1.48, incident_angle)
    footprint_wetsnow = func_footprint(snowheight, 1.81, incident_angle)
    footprint_verywetsnow = func_footprint(snowheight, 2.3, incident_angle)
    GNSS_refr_radius = pd.DataFrame(zip(snowheight * 100,
                                        footprint_drysnow * 100,
                                        footprint_moistsnow * 100,
                                        footprint_wetsnow * 100,
                                        footprint_verywetsnow * 100), columns=['h', 'r_ds', 'r_ms', 'r_ws', 'r_vws'])
    GNSS_refr_radius = GNSS_refr_radius.set_index(GNSS_refr_radius.h)
    return GNSS_refr_radius

def convert_swesh2density(swe, sh, rolling_median='3D'):
    """ calculate, calibrate, and filter snow density [kg/m3] from SWE and snow accumulation: density[kg/m3] = SWE [mm w.e.] * 1000 / sh[m])
    :param swe: dataframe containing swe values (in mm w.e.) from GNSS-refractometry
    :param sh: dataframe containing accumulation values (in m) from GNSS-reflectometry
    :param cal_date: date used for calibration (when 1m snow above antenna is reached)
    :param cal_val: calibration value from manual observations
    :return: density
    """
    # calculate density
    density = ((swe * 1000).divide(sh, axis=0)).dropna()

    # smoothen
    #density_fil = density.rolling(rolling_median).median()
    density_fil = density

    # remove densities lower than the density of new snow (50 kg/m3) or higher than the density of firn (830 kg/m3) or ice (917 kg/m3)
    density_cleaned = density_fil[~((density_fil < 50) | (density_fil >= 830))]

    return density_cleaned


def calc_new_snow_density(snow_height, swe, interval=6, min_acc=0.03):
    """
    calculate only the density of the freshly fallen snow
    :param snow_height: dataframe of the snow height of the entire snow pack over time
    :param swe: dataframe of the SWE of the entire snow pack over time
    :param interval: a number of days (must be an even number!) over which new snow accumulates and over which Delta SWE and Delta snow height are calculated
    :explanation:
    :new_snow: freshly accumulated snow layer
    :t2: quantity of the entire snowpack to a specific time
    :t1: quantity of the entire snowpack a specific time interval before (for example the day before)
    """
    # snow height of new layer
    h_t1 = snow_height.shift(int(interval/2))
    h_t2 = snow_height.shift(-int(interval/2))
    h_new_snow = h_t2 - h_t1
    h_new_snow[h_new_snow < min_acc] = np.nan

    # mass of new layer
    swe_t1 = swe.shift(int(interval/2))
    swe_t2 = swe.shift(-int(interval/2))
    swe_new_snow = swe_t2 - swe_t1

    # density of new layer
    d_new_snow = swe_new_snow/h_new_snow

    new_snow_datetimeindex, new_snow_heightindex = create_new_df(h_new_snow, 'h', swe_new_snow, 'swe', d_new_snow, 'density')

    return new_snow_datetimeindex, new_snow_heightindex

def convert_swe2sh(swe, ipol_density=None):
    """ calculate snow accumulation from swe: sh[m]  = SWE [mm w.e.] * 1000 / density[kg/m3]) using a mean density values or interpolated densities
    :param swe: dataframe containing swe values (in mm w.e.)
    :param ipol_density: use interpolated values, input interpolated densitiy values, otherwise=None: constant value is used
    :return: sh
    """
    if ipol_density is None:
        # calculate snow accumulation (sh) from SWE and a mean_density(0.5m)=408 from Hecht_2022
        sh = swe * 1000 / 408
    else:
        # calculate snow accumulation (sh) from SWE and interpolated density values (from manual Spuso observations)
        sh = ((swe * 1000).divide(ipol_density, axis=0)).dropna()

    return sh


def convert_sh2swe(sh, ipol_density=None):
    """ calculate swe from snow accumulation: swe[mm w.e.]  = (sh [mm] / 1000) * density[kg/m3]) using a mean density values or interpolated densities
    :param sh: dataframe containing snow accumulation (height) values (in meters)
    :param ipol_density: use interpolated values, input interpolated densitiy values, otherwise=None: constant value is used
    :return: swe
    """
    if ipol_density is None:
        # calculate SWE from snow accumulation (sh) and a mean_density(0.5m)=408 from Hecht_2022
        swe = (sh / 1000) * 408
    else:
        # calculate SWE from snow accumulation (sh) and interpolated density values (from manual Spuso observations)
        swe = ((sh / 1000).multiply(ipol_density, axis=0)).dropna()

    return swe


""" Define combined GNSS and reference sensors functions """


def convert_swe2sh_gnss(swe_gnss, ipol_density=None):
    """ convert GNSS derived SWE to snow accumulation using interpolated or a mean density value. Add SWE and converted sh to one dataframe
    :param swe_gnss: dataframe containing GNSS derived SWE estimations
    :param ipol_density: use interpolated values, input interpolated densitiy values, otherwise=None: constant value is used
    :return: gnss
    """
    print('\n-- convert GNSS SWE estimations to snow accumulation changes')
    sh_gnss = convert_swe2sh(swe_gnss, ipol_density)
    sh_gnss_const = convert_swe2sh(swe_gnss)

    # append new columns to existing gnss estimations dataframe
    gnss = pd.concat([swe_gnss, sh_gnss, sh_gnss_const], axis=1)
    gnss.columns = ['dswe', 'dsh', 'dsh_const']

    return gnss

def resample_gnss(gnss_leica, gnss_emlid, interval='D'):
    """ resample all sensors observations (different temporal resolutions) to other resolution
    :param gnss_leica: dataframe containing GNSS solutions (SWE, sh) from high-end system
    :param gnss_emlid: dataframe containing GNSS solutions (SWE, sh) from low-cost system
    :param interval: time interval for resampling, default=daily
    :return: leica_res, emlid_res
    """
    # resample sh and swe data (daily)
    if gnss_leica is not None:
        leica_res = (gnss_leica.resample(interval).median()).dropna()
    if gnss_emlid is not None:
        emlid_res = (gnss_emlid.resample(interval).median()).dropna()

    print('all gnss data is resampled with interval: %s' % interval)

    return leica_res, emlid_res

def resample_ref_obs(ref_df, interval='D'):
    """ resample all sensors observations (different temporal resolutions) to other resolution
    :param ref_df: dataframe containing observations (SWE, sh)
    :param interval: time interval for resampling, default=daily
    :return data_res: resampled dataframe
    """
    # resample sh and swe data (daily)
    data_res = (ref_df.resample(interval).median())
    print('reference data from %s is resampled with interval: %s' % (ref_df, interval))

    return data_res


def resample_allobs(gnss_leica, gnss_emlid, buoy=None, poles=None, laser=None, interval='D'):
    """ resample all sensors observations (different temporal resolutions) to other resolution
    :param gnss_leica: dataframe containing GNSS solutions (SWE, sh) from high-end system
    :param gnss_emlid: dataframe containing GNSS solutions (SWE, sh) from low-cost system
    :param buoy: dataframe containing snow buoy observations (SWE, sh)
    :param poles: dataframe containing poles observations (SWE, sh)
    :param laser: dataframe containing laser observations (SWE, sh)
    :param interval: time interval for resampling, default=daily
    :return: leica_res, emlid_res, buoy_res, poles_res, laser_res
    """
    # resample sh and swe data (daily)
    leica_res = (gnss_leica.resample(interval).median()).dropna()
    emlid_res = (gnss_emlid.resample(interval).median()).dropna()
    if buoy is not None:
        buoy_res = (buoy.resample(interval).median())
    if poles is not None:
        poles_res = (poles.resample(interval).median())
    if laser is not None:
        laser_res = (laser.resample(interval).median())

    print('all data is resampled with interval: %s' % interval)

    if buoy is not None and poles is not None and laser is not None:
        return leica_res, emlid_res, buoy_res, poles_res, laser_res
    elif buoy is None and poles is not None and laser is not None:
        return leica_res, emlid_res, poles_res, laser_res
    elif buoy is not None and poles is None and laser is not None:
        return leica_res, emlid_res, buoy_res, laser_res
    elif buoy is not None and poles is not None and laser is None:
        return leica_res, emlid_res, buoy_res, poles_res
    elif buoy is None and poles is None and laser is not None:
        return leica_res, emlid_res, laser_res
    elif buoy is not None and poles is None and laser is None:
        return leica_res, emlid_res, buoy_res
    elif buoy is None and poles is not None and laser is None:
        return leica_res, emlid_res, poles_res
    elif buoy is None and poles is None and laser is None:
        return leica_res, emlid_res


def get_mean_and_std_deviation(df_poles, df_buoy):
    """
    :param df_poles: DataFrame of pole observations containing snow height measurements (of 16 stakes) named '1', '2', '3',... and SWE calculations named 'dswe1', ...
    _param df_buoy: DataFrame of buoy observations containing snow height measurements, dsh and dswe (of 4 buoys)
    """
    # Calculate mean and standard deviation of snow height measurements (shm) and dSWE
    # of stake field (pole) measurements
    df_poles['sh_mean'] = df_poles[
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']].mean(axis=1)
    df_poles['sh_std'] = df_poles[
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']].std(axis=1)
    df_poles['dswe_mean'] = df_poles[
        ['dswe1', 'dswe2', 'dswe3', 'dswe4', 'dswe5', 'dswe6', 'dswe7', 'dswe8', 'dswe9', 'dswe10', 'dswe11', 'dswe12',
         'dswe13', 'dswe14', 'dswe15', 'dswe16']].mean(axis=1)
    df_poles['dswe_std'] = df_poles[
        ['dswe1', 'dswe2', 'dswe3', 'dswe4', 'dswe5', 'dswe6', 'dswe7', 'dswe8', 'dswe9', 'dswe10', 'dswe11', 'dswe12',
         'dswe13', 'dswe14', 'dswe15', 'dswe16']].std(axis=1)
    # of buoy measurements
    df_buoy['sh_mean'] = df_buoy[['sh1', 'sh2', 'sh3', 'sh4']].mean(axis=1)
    df_buoy['sh_std'] = df_buoy[['sh1', 'sh2', 'sh3', 'sh4']].std(axis=1)
    df_buoy['dsh_mean'] = df_buoy[['dsh1', 'dsh2', 'dsh3', 'dsh4']].mean(axis=1)
    df_buoy['dsh_std'] = df_buoy[['dsh1', 'dsh2', 'dsh3', 'dsh4']].std(axis=1)
    df_buoy['dswe_mean'] = df_buoy[['dswe1', 'dswe2', 'dswe3', 'dswe4']].mean(axis=1)
    df_buoy['dswe_std'] = df_buoy[['dswe1', 'dswe2', 'dswe3', 'dswe4']].std(axis=1)

    return df_poles, df_buoy


def calculate_stats(title, reference_data, data, unit='kg/m2'):
    """
    calculate Residuals/difference/deviation of data from a reference, deviation range, variance, standard deviation (RMSE) and R²
    :param reference_data: array data series or model from which the deviation shall be calculated
    :param data: array data series of which the deviation from the reference series is calculated
    """
    # calculate residuals (Abweichung der Daten von der ermittelten Funktion)
    R = data - reference_data
    R_p = ((data - reference_data)/reference_data)*100
    # number of samples (eg number of observations made on the same day)
    len = R.shape[0]
    # mean deviation (mean of all residuals)
    ME = np.mean(R)
    ME_p = np.mean(R_p)
    # mean deviation (mean of all residuals)
    MAE = np.mean(np.abs(R))
    MAE_p = np.mean(np.abs(R_p))
    # calculate maximum range of deviation from function
    range = np.max(R) - np.min(R)
    range_p = np.max(R_p) - np.min(R_p)
    # variance (mean of squared residuals/error)
    MSE = np.mean(R**2)
    MSE_p = np.mean(R_p**2)
    # root mean square error (residuals) = standard deviation
    RMSE = MSE**0.5
    RMSE_p = MSE_p**0.5
    # standard deviation / standard error
    std = np.std(R)
    std_p = np.std(R_p)
    # sum of squared residuals
    SSR = np.sum(R**2)
    # Total sum of squared residuals (totale Varianz der y-daten)
    TSS = np.sum((data-np.mean(data))**2)
    # calculate R²
    R_squared = 1 - (SSR / TSS)

    print('\n=========================================')
    print('Statistics: %s' % title)
    print('number of data points/ samples: %s' % len)
    print('R_squared: %s' % R_squared)
    print('maximum range of deviation from curve: '
          '%s %s (%s %%)' % (range, unit, range_p))
    print('mean deviation/error (ME):'
          '%s %s (%s %%)' % (ME, unit, ME_p))
    print('mean of absolute values of deviation:'
          '%s %s (%s %%)' % (MAE, unit, MAE_p))
    print('Variance: Mean Squared Error (MSE):'
          '%s %s (%s %%)' % (MSE, unit, MSE_p))
    print('standard deviation (with formula calculated) - Root Mean Squared Error (RMSE):'
          '%s %s (%s %%)' % (RMSE, unit, RMSE_p))
    print('standard deviation (with python np.std()):'
          '%s %s (%s %%)' % (std, unit, std_p))
    print('=========================================')

    return R, R_p


def dependency(x_data, x_data_name, y_data, y_data_name, create_plot, fig_name, save_plot, dest_path, plot_fit = True, unit='kg/m²', fig_title='Snow Height', ax_lim=(0, 150), tick_interval=20, color='darkblue'):
    """
    merge two time series into one dataframe, calculate linear fit, statistics and plot
    :param x_data: reference data
    :param y_data: data series
    """

    print('\nDependence of %s and %s' % (x_data_name, y_data_name))

    # merge data frame
    dataframe, df_x_index = create_new_df(x_data, x_data_name, y_data, y_data_name)

    # calculate linear fit - linear regression
    x_y_fit = calculate_linearfit(dataframe, x_data_name, y_data_name)

    # calculate deviation of y-data from linear fit
    R, R_p = calculate_stats('Deviation between %s and linear fit' % y_data_name, x_y_fit[1], np.array(dataframe[y_data_name]), unit)
    R, R_p = calculate_stats('Deviation between %s and %s' % (y_data_name, x_data_name), dataframe[x_data_name], dataframe[y_data_name], unit)

    # plot y-data over x_data with fit
    if create_plot is True and plot_fit is True:
        plot_ds(dest_path,
                fig_name,
                fig_title=fig_title,
                create_plot=create_plot,
                save=save_plot,
                fig_size=(7, 7),
                x_datetime=False,
                x_lim=ax_lim,
                y_lim=ax_lim,
                xtick_interval=tick_interval,
                ytick_interval=tick_interval,
                y_label=y_data_name,
                x_label=x_data_name,
                ds1_fit=x_y_fit,
                ds1_color='lightgrey',
                ds1_fit_label='fit: {:.3f}x + {:.3f}'.format(*x_y_fit[2]),
                ds6=df_x_index,
                ds6_color=color,
                ds6_label=None)
    elif create_plot is True and plot_fit is False:
        plot_ds(dest_path,
                fig_name,
                fig_title=fig_title,
                create_plot=create_plot,
                save=save_plot,
                fig_size=(7, 7),
                x_datetime=False,
                x_lim=ax_lim,
                y_lim=ax_lim,
                xtick_interval=tick_interval,
                ytick_interval=tick_interval,
                y_label=y_data_name,
                x_label=x_data_name,
                ds6=df_x_index,
                ds6_color=color,
                ds6_label=None)

    return R, R_p

def create_new_df(x_data, x_data_name, y_data, y_data_name, y_data2=None, y_data2_name=None, y_data3=None, y_data3_name=None, y_data4=None, y_data4_name=None):
    """
    creates a new dataframe from two dataseries, with the same datetime index, setting x_data to new index
    :param x_data: time series with same index as y_data
    :param y_data: time series with same index as x_data
    :param x_data_name: name of x-data
    :param y_data_name: name of y-data
    """
    if y_data2 is None and y_data3 is None and y_data4 is None:
        dataframe = pd.concat([x_data, y_data], axis=1)
        dataframe = dataframe.dropna()
        dataframe.columns = [x_data_name, y_data_name]
        df_x_index = dataframe.set_index(x_data_name)
    elif y_data3 is None and y_data4 is None:
        dataframe = pd.concat([x_data, y_data, y_data2], axis=1)
        dataframe.columns = [x_data_name, y_data_name, y_data2_name]
        dataframe = dataframe.dropna()
        df_x_index = dataframe.set_index(x_data_name)
    elif y_data4 is None:
        dataframe = pd.concat([x_data, y_data, y_data2, y_data3], axis=1)
        dataframe.columns = [x_data_name, y_data_name, y_data2_name, y_data3_name]
        dataframe = dataframe.dropna()
        df_x_index = dataframe.set_index(x_data_name)
    else:
        dataframe = pd.concat([x_data, y_data, y_data2, y_data3, y_data4], axis=1)
        dataframe.columns = [x_data_name, y_data_name, y_data2_name, y_data3_name, y_data4_name]
        dataframe = dataframe.dropna()
        df_x_index = dataframe.set_index(x_data_name)
    return dataframe, df_x_index


def calculate_linearfit(dataframe, x_data_name, y_data_name):
    """"
    :param dataframe: dataframe containing y-data as index and x-data as first column
    :return: 3D array with array[0]=x-values, array[1]=y-values of fit, array[2]=fit parameters
    """
    x_data = dataframe[x_data_name]
    y_data = dataframe[y_data_name]

    # calculate linear fit
    fit_param = np.polyfit(x_data, y_data, 1)
    print('Linear fit:'
          'm =', fit_param[0].round(2),
          ',b =', fit_param[1].round(3))

    # array of y-data of linear fit
    data_fit = func_linear(x_data, fit_param[0], fit_param[1])
    # create a 3D array for plotting the fitted curve by add x-values and function parameters
    # -> array[0]=x-values, array[1]=y-values, array[2]=function params
    x_y_fit = [np.array(x_data), data_fit, fit_param]

    return x_y_fit


def solution_control(amb_state, std_gnss_daily, std_guess):

    # manual fit
    x_values = np.linspace(1, 96, 951)
    func_exp = func_exp3(x_values, std_guess)
    func_exp = [x_values, func_exp, std_guess]

    nr_fixed = amb_state[(amb_state == 1)].resample('D').count()
    amb_std_datetime, amb_std = create_new_df(nr_fixed, 'Number of fixed ambiguities', std_gnss_daily, 'Daily Noise')
    amb_std = amb_std.sort_values(by='Number of fixed ambiguities')
    # exponential regression
    amb_std_fit = exponential_regression('Daily Noise over Nb solutions', amb_std.index, amb_std['Daily Noise'], function='std/sqrt(n)', guess=(30))

    return amb_std, amb_std_fit, func_exp





""" Define linear and exponential functions """

def func_linear(x, m, b):
    return m * x + b

def func_footprint(h, n, a):
    """
    h: snowheight
    n: refractive index of snow
    a: angle of incidence
    """
    return h*(math.tan(math.asin(math.sin(math.radians(a))/n)))


def func_err_prop_single(a, b, c, m_err, h_err):
    h = np.linspace(0, 10, 1001)
    m = func_exp(h, a, b, c)
    density = m / h
    rel_err = np.sqrt((m_err/m)**2+(h_err/h)**2)
    abs_err = rel_err * density
    error = pd.DataFrame(zip(h, rel_err, abs_err), columns=['h', 'rel_m'+str(m_err)+'_h'+''.join(str(h_err).split('.')), 'abs_m'+str(m_err)+'_h'+''.join(str(h_err).split('.'))])
    error = error.set_index(error.h)
    return error


def func_err_prop(a, b, c, m_err1, m_err2, m_err3, m_err4, h_err1, h_err2, h_err3, h_err4):
    error1 = func_err_prop_single(a, b, c, m_err1, h_err1)
    error2 = func_err_prop_single(a, b, c, m_err2, h_err2)
    error3 = func_err_prop_single(a, b, c, m_err3, h_err3)
    error4 = func_err_prop_single(a, b, c, m_err4, h_err4)
    error = pd.concat([error1, error2, error3, error4], axis=1)
    return error


def func_err_prop2(a, b, c, m_err, h_err, density1, density2, density3, density4):
    # define h and m
    h = np.linspace(0, 10, 1001)
    m = func_exp(h, a, b, c)
    # caluclate relative error
    rel_err = np.sqrt((m_err/m)**2+(h_err/h)**2)
    # calculate absolute error
    abs_err1 = rel_err * density1
    abs_err2 = rel_err * density2
    abs_err3 = rel_err * density3
    abs_err4 = rel_err * density4
    # merge into one dataframe
    error = pd.DataFrame(zip(h, rel_err, abs_err1, abs_err2, abs_err3, abs_err4), columns=['h', 'rel_err', 'abs_err_'+str(density1), 'abs_err_'+str(density2), 'abs_err_'+str(density3), 'abs_err_'+str(density4)])
    error = error.set_index(error.h)
    return error


def func_exp(x, a, b, c):
    return a * np.exp(b * x) + c

def func_exp2(x, a, b):
    return a * np.exp(b * x)

def func_exp3(x, std):
    return std/np.sqrt(x)

def exponential_regression(title, x_data, y_data, function='a * exp(b * x) + c', y_0=1, guess=(100, 1, -100), unit='kg/m2'):
    """
    :param x_data: array of x-data
    :param y_data: array of y-data
    """
    if function == 'a * exp(b * x) + c':
        sigma = np.ones(len(x_data))
        if y_0 == 0:
            sigma[0] = 0.000001  # fit is going through first point
        popt, pcov = curve_fit(func_exp, x_data, y_data, p0=guess, sigma=sigma)
        # y-values of fit
        fit, popt = func_exp(x_data, *popt), popt
        # create a 3D array with array[0]=x-values, array[1]=y-values of fit, array[2]=function params
        exp_x_fit_params = [x_data, fit, popt]

        print('\n=========================================')
        print('Exponential Regression -  %s' % title)
        print('optimized parameters of function = a * exp(b * x) + c' % popt)
        print('a       = %s' % popt[0])
        print('sigma a = %s' % np.sqrt(np.diag(pcov))[0])
        print('b       = %s' % popt[1])
        print('sigma b = %s' % np.sqrt(np.diag(pcov))[1])
        print('c       = %s' % popt[2])
        print('sigma c = %s' % np.sqrt(np.diag(pcov))[2])
        print('=========================================')

        # calculate R², variance (MSE), and standard deviation (RMSE) between y_data and fit
        calculate_stats(title, fit, y_data, unit)

        return exp_x_fit_params
    elif function == 'a * exp(b * x)':
        popt, pcov = curve_fit(func_exp2, x_data, y_data, p0=guess)
        # y-values of fit
        fit, popt = func_exp2(x_data, *popt), popt
        # create a 3D array with array[0]=x-values, array[1]=y-values of fit, array[2]=function params
        exp_x_fit_params = [x_data, fit, popt]

        print('\n=========================================')
        print('Exponential Regression -  %s' % title)
        print('optimized parameters of function = a * exp(b * x)' % popt)
        print('a       = %s' % popt[0])
        print('sigma a = %s' % np.sqrt(np.diag(pcov))[0])
        print('b       = %s' % popt[1])
        print('sigma b = %s' % np.sqrt(np.diag(pcov))[1])
        print('=========================================')

        # calculate R², variance (MSE), and standard deviation (RMSE) between y_data and fit
        calculate_stats(title, fit, y_data, unit)

        return exp_x_fit_params
    elif function == 'std/sqrt(n)':
        popt, pcov = curve_fit(func_exp3, x_data, y_data, p0=guess)
        # y-values of fit
        fit, popt = func_exp3(x_data, *popt), popt
        # create a 3D array with array[0]=x-values, array[1]=y-values of fit, array[2]=function params
        exp_x_fit_params = [x_data, fit, popt]

        print('\n=========================================')
        print('Exponential Regression -  %s' % title)
        print('optimized parameters of function std(n) = std1 / sqrt(n)' % popt)
        print('std1       = %s' % popt[0])
        print('sigma std1 = %s' % np.sqrt(np.diag(pcov))[0])
        print('=========================================')

        # calculate R², variance (MSE), and standard deviation (RMSE) between y_data and fit
        calculate_stats(title, fit, y_data, unit)

        return exp_x_fit_params
    else:
        print('\nExponential regression was not executed, function must be specified to either "a * exp(b * x)" or "a * exp(b * x) + c"!'
              'Fit and parameters are defined to zero')
        exp_x_fit_params = [0, 0, 0]
        return exp_x_fit_params




""" Define plot functions """

def plot_ds(data_path, plot_name, create_plot=True, save=[False, True],
            ds1=None, ds1_bar=None, ds1_std=None, ds1_bias=None, ds1_err=None, ds1_fit=None, ds1_yaxis=1, ds1_linestyle='-', ds1_fit_linestyle='-', ds1_transparency=0.4, ds1_marker=' ', ds1_markersize=4, ds1_color='crimson', ds1_label='Leica Base to Leica Rover (LBLR)', ds1_std_label='daily noise', ds1_fit_label=' ',
            ds2=None, ds2_bar=None, ds2_std=None, ds2_bias=None, ds2_err=None, ds2_fit=None, ds2_yaxis=1, ds2_linestyle='-', ds2_fit_linestyle='-', ds2_transparency=0.4, ds2_marker=' ', ds2_markersize=4, ds2_color='salmon', ds2_label='Leica Base to ublox Rover (LBUR)', ds2_std_label='daily noise', ds2_fit_label=' ',
            ds3=None, ds3_bar=None, ds3_std=None, ds3_bias=None, ds3_err=None, ds3_fit=None, ds3_yaxis=1, ds3_linestyle='-', ds3_fit_linestyle='-', ds3_transparency=0.4, ds3_marker=' ', ds3_markersize=4, ds3_color='dodgerblue', ds3_label='JAVAD Base to Leica Rover (JBLR)', ds3_std_label='daily noise', ds3_fit_label=' ',
            ds4=None, ds4_bar=None, ds4_std=None, ds4_bias=None, ds4_err=None, ds4_fit=None, ds4_yaxis=1, ds4_linestyle='-', ds4_fit_linestyle='-', ds4_transparency=0.4, ds4_marker=' ', ds4_markersize=4, ds4_color='deepskyblue', ds4_label='JAVAD Base to ublox Rover (JBUR)', ds4_std_label='daily noise', ds4_fit_label=' ',
            ds5=None, ds5_bar=None, ds5_std=None, ds5_bias=None, ds5_err=None, ds5_fit=None, ds5_yaxis=1, ds5_linestyle=' ', ds5_fit_linestyle='-', ds5_transparency=0, ds5_marker='+', ds5_markersize=10, ds5_color='k', ds5_label='snow pit', ds5_std_label='error', ds5_fit_label=' ',
            ds6=None, ds6_bar=None, ds6_std=None, ds6_bias=None, ds6_err=None, ds6_fit=None, ds6_yaxis=1, ds6_linestyle=' ', ds6_fit_linestyle='-', ds6_transparency=0, ds6_marker='o', ds6_markersize=4, ds6_color='darkseagreen', ds6_label='laser (shm)', ds6_std_label='daily noise', ds6_fit_label=' ',
            ds7=None, ds7_bar=None, ds7_std=None, ds7_bias=None, ds7_err=None, ds7_fit=None, ds7_yaxis=1, ds7_linestyle=' ', ds7_fit_linestyle='-', ds7_transparency=0, ds7_marker='o', ds7_markersize=4, ds7_color='darkgray', ds7_label='buoy', ds7_std_label='σ', ds7_fit_label=' ',
            ds8=None, ds8_bar=None, ds8_std=None, ds8_bias=None, ds8_err=None, ds8_fit=None, ds8_yaxis=1, ds8_linestyle=':', ds8_fit_linestyle='-', ds8_transparency=0, ds8_marker=' ', ds8_markersize=4, ds8_color='darkkhaki', ds8_label='stake field', ds8_std_label='σ', ds8_fit_label=' ',
            ds9=None, ds9_bar=None, ds9_std=None, ds9_bias=None, ds9_err=None, ds9_fit=None, ds9_yaxis=2, ds9_linestyle='-', ds9_fit_linestyle='-', ds9_transparency=0, ds9_marker=' ', ds9_markersize=4, ds9_color='goldenrod', ds9_label='GNSS-IR', ds9_std_label='daily noise', ds9_fit_label=' ',
            plot_date_lines=False, plot_vline=None, hline_value=None, ax2_hline_value=None, hline_label=None, suffix='', base_abb=None,
            fig_title=None, title_size=22, fig_size=(15, 9),
            x_datetime=True, x_locator='month', major_day_locator=5, x_stepsize = 1, xtick_rotation=45,
            switch_xy=False, invert_y1axis=False,
            x_label=None, x_lim=None, xtick_interval=100,
            y_label='mass [kg/m²]', legend_position=(0.01, 0.88), legend_title=None, y_lim=None, ytick_interval=100,
            y_axis2 = False, y2_label='snow height [cm]', legend2_position=(0.8, 0.2), legend2_title=None, y2_lim=None, y2tick_interval=100,
            y_axis3 = False, y3_label='density [kg/m³]', legend3_position=(0.8, 0.88), legend3_title=None, y3_lim=None, y3tick_interval=50,
            axes_labelsize=20, xtick_labelsize=18, ytick_labelsize=18,
            legend_fontsize=18, lines_linewidth=1.7, bar_width=0.8):
    """
    Plot data (time) series (ds) with deviation / error bars
    :param ds:      data series in form of a float containing x-values (Datetime) as index and
                    y-values as one column (but no more columns!)
    :param ds_std:  standard deviation of data series (same format as ds)
    :param ds_err:  deviation from data points - for error bars (same format as ds)
    :param ds_fit:  Fit line or curve (of regression analysis) need to be handed as 3D-array [x_data, y_data, fit_parameters]
    :param ds_label/linestyle/marker/color: plot specifications for each dataseires
    :param save:    if True plot will be saved in processing_directory/30_plots folder
                    is False plot will be shown
    :param base_abb: Abbreviation of shown GNSS-base data (LB or JB) (if only one is selected, data will be saved in a separate folder)
    """

    plt.close()

    if create_plot == True:
        print('\nPlotting figure: %s' % plot_name)

        # Q: set plot parameters
        plt.rcParams['axes.labelsize'] = axes_labelsize
        plt.rcParams['xtick.labelsize'] = xtick_labelsize
        plt.rcParams['ytick.labelsize'] = ytick_labelsize
        plt.rcParams['legend.fontsize'] = legend_fontsize
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['lines.linewidth'] = lines_linewidth

        # Q: create figure
        fig, ax1 = plt.subplots(figsize=fig_size, constrained_layout=True)
        if y_axis2 == True:
            ax2 = ax1.twinx()
        if y_axis3 == True:
            ax3 = ax1.twinx()

        # Q: create dictionaries to grap parameters
        dict_ds = {1: ds1, 2: ds2, 3: ds3, 4: ds4, 5: ds5, 6: ds6, 7: ds7, 8: ds8, 9: ds9}
        dict_std = {1: ds1_std, 2: ds2_std, 3: ds3_std, 4: ds4_std, 5: ds5_std, 6: ds6_std, 7: ds7_std, 8: ds8_std, 9: ds9_std}
        dict_std_label = {1: ds1_std_label, 2: ds2_std_label, 3: ds3_std_label, 4: ds4_std_label, 5: ds5_std_label, 6: ds6_std_label, 7: ds7_std_label, 8: ds8_std_label, 9: ds9_std_label}
        dict_err = {1: ds1_err, 2: ds2_err, 3: ds3_err, 4: ds4_err, 5: ds5_err, 6: ds6_err, 7: ds7_err, 8: ds8_err, 9: ds9_err}
        dict_label = {1: ds1_label, 2: ds2_label, 3: ds3_label, 4: ds4_label, 5: ds5_label, 6: ds6_label, 7: ds7_label, 8: ds8_label, 9: ds9_label}
        dict_linestyle = {1: ds1_linestyle, 2: ds2_linestyle, 3: ds3_linestyle, 4: ds4_linestyle, 5: ds5_linestyle, 6: ds6_linestyle, 7: ds7_linestyle, 8: ds8_linestyle, 9: ds9_linestyle}
        dict_marker = {1: ds1_marker, 2: ds2_marker, 3: ds3_marker, 4: ds4_marker, 5: ds5_marker, 6: ds6_marker, 7: ds7_marker, 8: ds8_marker, 9: ds9_marker}
        dict_markersize = {1: ds1_markersize, 2: ds2_markersize, 3: ds3_markersize, 4: ds4_markersize, 5: ds5_markersize, 6: ds6_markersize, 7: ds7_markersize, 8: ds8_markersize, 9: ds9_markersize}
        dict_color = {1: ds1_color, 2: ds2_color, 3: ds3_color, 4: ds4_color, 5: ds5_color, 6: ds6_color, 7: ds7_color, 8: ds8_color, 9: ds9_color}
        dict_transparency = {1: ds1_transparency, 2: ds2_transparency, 3: ds3_transparency, 4: ds4_transparency, 5: ds5_transparency, 6: ds6_transparency, 7: ds7_transparency, 8: ds8_transparency, 9: ds9_transparency}
        dict_yaxis = {1: ds1_yaxis, 2: ds2_yaxis, 3: ds3_yaxis, 4: ds4_yaxis, 5: ds5_yaxis, 6: ds6_yaxis, 7: ds7_yaxis, 8: ds8_yaxis, 9: ds9_yaxis}
        dict_fit = {1: ds1_fit, 2: ds2_fit, 3: ds3_fit, 4: ds4_fit, 5: ds5_fit, 6: ds6_fit, 7: ds7_fit, 8: ds8_fit, 9: ds9_fit}
        dict_fit_label = {1: ds1_fit_label, 2: ds2_fit_label, 3: ds3_fit_label, 4: ds4_fit_label, 5: ds5_fit_label, 6: ds6_fit_label, 7: ds7_fit_label, 8: ds8_fit_label, 9: ds9_fit_label}
        dict_fit_linestyle = {1: ds1_fit_linestyle, 2: ds2_fit_linestyle, 3: ds3_fit_linestyle, 4: ds4_fit_linestyle, 5: ds5_fit_linestyle, 6: ds6_fit_linestyle, 7: ds7_fit_linestyle, 8: ds8_fit_linestyle, 9: ds9_fit_linestyle}
        dict_bar = {1: ds1_bar, 2: ds2_bar, 3: ds3_bar, 4: ds4_bar, 5: ds5_bar, 6: ds6_bar, 7: ds7_bar, 8: ds8_bar, 9: ds9_bar}
        dict_bias = {1: ds1_bias, 2: ds2_bias, 3: ds3_bias, 4: ds4_bias, 5: ds5_bias, 6: ds6_bias, 7: ds7_bias, 8: ds8_bias, 9: ds9_bias}

        # Q: plotting all handed data
        for i in range(1, 9 + 1):
            if switch_xy == False:
                # Q: plot data series (ds)
                if dict_yaxis[i] == 1:
                    if dict_ds[i] is not None:
                        ax1.plot(dict_ds[i].index, dict_ds[i],
                                 label=dict_label[i],
                                 linestyle=dict_linestyle[i],
                                 marker=dict_marker[i],
                                 markersize=dict_markersize[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: plot standard deviation of data series (_std)
                    if dict_std[i] is not None:
                        ax1.fill_between(dict_ds[i].index, dict_ds[i] - dict_std[i], dict_ds[i] + dict_std[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: plot bias of data series (_bias)
                    if dict_bias[i] is not None:
                        ax1.fill_between(dict_ds[i].index, dict_ds[i], dict_ds[i] + dict_bias[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: Plot error bars
                    if dict_err[i] is not None:
                        ax1.errorbar(dict_err[i].index, dict_ds[i],
                                     yerr=dict_err[i],
                                     color='k',
                                     linestyle='',
                                     capsize=4,
                                     alpha=0.5)
                    if dict_bar[i] is not None:
                        ax1.bar(dict_bar[i].index, dict_bar[i],
                                 width=bar_width,
                                 label=dict_label[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: fit/ regression curve
                    if dict_fit[i] is not None:
                        ax1.plot(dict_fit[i][0], dict_fit[i][1],
                                 linestyle=dict_fit_linestyle[i],
                                 color=dict_color[i],
                                 label=dict_fit_label[i])

                if dict_yaxis[i] == 2:
                    if dict_ds[i] is not None:
                        ax2.plot(dict_ds[i].index, dict_ds[i],
                                 label=dict_label[i],
                                 linestyle=dict_linestyle[i],
                                 marker=dict_marker[i],
                                 markersize=dict_markersize[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: plot standard deviation of data series (_std)
                    if dict_std[i] is not None:
                        ax2.fill_between(dict_ds[i].index, dict_ds[i] - dict_std[i], dict_ds[i] + dict_std[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: plot bias of data series (_bias)
                    if dict_bias[i] is not None:
                        ax1.fill_between(dict_ds[i].index, dict_ds[i], dict_ds[i] + dict_bias[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: Plot error bars
                    if dict_err[i] is not None:
                        ax2.errorbar(dict_err[i].index, dict_ds[i],
                                     yerr=dict_err[i],
                                     color='k',
                                     linestyle='',
                                     capsize=4,
                                     alpha=0.5)
                    if dict_bar[i] is not None:
                        ax2.bar(dict_bar[i].index, dict_bar[i],
                                 width=bar_width,
                                 label=dict_label[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: Plot exponential regression curve
                    if dict_fit[i] is not None:
                        ax2.plot(dict_fit[i][0], dict_fit[i][1],
                                 linestyle=dict_fit_linestyle[i],
                                 color=dict_color[i],
                                 label=dict_fit_label[i])

                if dict_yaxis[i] == 3:
                    if dict_ds[i] is not None:
                        ax3.plot(dict_ds[i].index, dict_ds[i],
                                 label=dict_label[i],
                                 linestyle=dict_linestyle[i],
                                 marker=dict_marker[i],
                                 markersize=dict_markersize[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: plot standard deviation of data series (_std)
                    if dict_std[i] is not None:
                        ax3.fill_between(dict_ds[i].index, dict_ds[i] - dict_std[i], dict_ds[i] + dict_std[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: plot bias of data series (_bias)
                    if dict_bias[i] is not None:
                        ax3.fill_between(dict_ds[i].index, dict_ds[i], dict_ds[i] + dict_bias[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: Plot error bars
                    if dict_err[i] is not None:
                        ax3.errorbar(dict_err[i].index, dict_ds[i],
                                     yerr=dict_err[i],
                                     color='k',
                                     linestyle='',
                                     capsize=4,
                                     alpha=0.5)
                    if dict_bar[i] is not None:
                        ax3.bar(dict_bar[i].index, dict_bar[i],
                                 width=bar_width,
                                 label=dict_label[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: Plot exponential regression curve
                    if dict_fit[i] is not None:
                        ax3.plot(dict_fit[i][0], dict_fit[i][1],
                                 linestyle=dict_fit_linestyle[i],
                                 color=dict_color[i],
                                 label=dict_fit_label[i])

            if switch_xy == True:
                # Q: plot data series (ds)
                if dict_yaxis[i] == 1:
                    if dict_ds[i] is not None:
                        ax1.plot(dict_ds[i], dict_ds[i].index,
                                 label=dict_label[i],
                                 linestyle=dict_linestyle[i],
                                 marker=dict_marker[i],
                                 markersize=dict_markersize[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: plot standard deviation of data series (_std)
                    if dict_std[i] is not None:
                        ax1.fill_between(dict_ds[i] - dict_std[i], dict_ds[i] + dict_std[i], dict_ds[i].index,
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: plot bias of data series (_bias)
                    if dict_bias[i] is not None:
                        ax1.fill_between(dict_ds[i].index, dict_ds[i], dict_ds[i] + dict_bias[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: Plot error bars
                    if dict_err[i] is not None:
                        ax1.errorbar(dict_err[i], dict_ds[i],
                                     yerr=dict_err[i],
                                     color='k',
                                     linestyle='',
                                     capsize=4,
                                     alpha=0.5)
                    # Q: Plot exponential regression curve
                    if dict_fit[i] is not None:
                        ax1.plot(dict_fit[i][1], dict_fit[i][0],
                                 linestyle=dict_fit_linestyle[i],
                                 color=dict_color[i],
                                 label=dict_fit_label[i])

                if dict_yaxis[i] == 2:
                    if dict_ds[i] is not None:
                        ax2.plot(dict_ds[i], dict_ds[i].index,
                                 label=dict_label[i],
                                 linestyle=dict_linestyle[i],
                                 marker=dict_marker[i],
                                 markersize=dict_markersize[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: plot standard deviation of data series (_std)
                    if dict_std[i] is not None:
                        ax2.fill_between(dict_ds[i] - dict_std[i], dict_ds[i] + dict_std[i], dict_ds[i].index,
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: plot bias of data series (_bias)
                    if dict_bias[i] is not None:
                        ax1.fill_between(dict_ds[i].index, dict_ds[i], dict_ds[i] + dict_bias[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: Plot error bars
                    if dict_err[i] is not None:
                        ax2.errorbar(dict_err[i], dict_ds[i],
                                     yerr=dict_err[i],
                                     color='k',
                                     linestyle='',
                                     capsize=4,
                                     alpha=0.5)
                    # Q: Plot exponential regression curve
                    if dict_fit[i] is not None:
                        ax2.plot(dict_fit[i][1], dict_fit[i][0],
                                 linestyle=dict_fit_linestyle[i],
                                 color=dict_color[i],
                                 label=dict_fit_label[i])

                if dict_yaxis[i] == 3:
                    if dict_ds[i] is not None:
                        ax3.plot(dict_ds[i].index, dict_ds[i],
                                 label=dict_label[i],
                                 linestyle=dict_linestyle[i],
                                 marker=dict_marker[i],
                                 markersize=dict_markersize[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: plot standard deviation of data series (_std)
                    if dict_std[i] is not None:
                        ax3.fill_between(dict_ds[i].index, dict_ds[i] - dict_std[i], dict_ds[i] + dict_std[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: plot bias of data series (_bias)
                    if dict_bias[i] is not None:
                        ax3.fill_between(dict_ds[i].index, dict_ds[i], dict_ds[i] + dict_bias[i],
                                         color=dict_color[i],
                                         alpha=0.2,
                                         label=dict_std_label[i])
                    # Q: Plot error bars
                    if dict_err[i] is not None:
                        ax3.errorbar(dict_err[i].index, dict_ds[i],
                                     yerr=dict_err[i],
                                     color='k',
                                     linestyle='',
                                     capsize=4,
                                     alpha=0.5)
                    if dict_bar[i] is not None:
                        ax3.bar(dict_bar[i].index, dict_bar[i],
                                 width=bar_width,
                                 label=dict_label[i],
                                 color=dict_color[i],
                                 alpha=1-dict_transparency[i])
                    # Q: Plot exponential regression curve
                    if dict_fit[i] is not None:
                        ax3.plot(dict_fit[i][0], dict_fit[i][1],
                                 linestyle=dict_fit_linestyle[i],
                                 color=dict_color[i],
                                 label=dict_fit_label[i])


        # Q: figure annotations
        fig.suptitle(fig_title, y=0.93, fontsize=title_size)
        leg = ax1.legend(title=legend_title, loc='upper left', bbox_to_anchor=legend_position, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        for line in leg.get_lines():
            line.set_linewidth(2.5)
            line.set_markersize(8)

        # Q: axis settings
        ax1.set_xlabel(x_label)
        ax1.grid(color='lightgrey', linestyle='-', linewidth=1, alpha=0.5)
        # Q: x-axis
        ax1.set_xlim(x_lim)
        if x_datetime == True:
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks(np.arange(start, end, x_stepsize))
            if x_locator == 'day':
                ax1.xaxis.set_major_locator(DayLocator(interval=major_day_locator))
                ax1.xaxis.set_minor_locator(AutoMinorLocator(x_stepsize))
            else:
                ax1.xaxis.set_major_locator(MonthLocator())
                ax1.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
        else:
            ax1.xaxis.set_major_locator(MultipleLocator(xtick_interval))
            ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        if y_axis2 is False:
            plt.xticks(rotation=xtick_rotation)
        if plot_date_lines is True:
            plt.axvline(dt.datetime(2021, 11, 16), color='red')
            plt.axvline(dt.datetime(2021, 11, 27), color='goldenrod')
            plt.axvline(dt.datetime(2021, 12, 21), color='dodgerblue')
        if plot_vline is not None:
            plt.axvline(plot_vline, color='grey')
        if hline_value is not None:
            ax1.axhline(y=hline_value, color='grey', label=hline_label)
        if ax2_hline_value is not None:
            ax2.axhline(y=ax2_hline_value, color='grey', label=hline_label)

        # Q: y-axis
        ax1.set_ylabel(y_label)
        ax1.set_ylim(y_lim)
        ax1.yaxis.set_major_locator(MultipleLocator(ytick_interval))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
        if invert_y1axis == True:
            ax1.invert_yaxis()
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position('top')

        if y_axis2 == True:
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax2.set_ylabel(y2_label)
            ax2.set_ylim(y2_lim)
            ax2.yaxis.set_major_locator(MultipleLocator(y2tick_interval))
            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
            leg2 = ax2.legend(title=legend2_title, loc='upper left', bbox_to_anchor=legend2_position, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            for line in leg2.get_lines():
                line.set_linewidth(2.5)
                line.set_markersize(8)

        if y_axis3 == True:
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax3.set_ylabel(y3_label)
            ax3.set_ylim(y3_lim)
            ax3.yaxis.set_major_locator(MultipleLocator(y3tick_interval))
            ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
            leg3 = ax3.legend(title=legend3_title, loc='upper left', bbox_to_anchor=legend3_position, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            rspine = ax3.spines['right']
            rspine.set_position(('axes', 1.1))
            for line in leg3.get_lines():
                line.set_linewidth(2.5)
                line.set_markersize(8)

        # Q: Options: Save figure
        if save is True and base_abb is not None:
            plt.savefig(data_path + '/30_plots/' + base_abb + plot_name + suffix + '.png')
            plt.savefig(data_path + '/30_plots/' + base_abb + plot_name + suffix + '.pdf')
            print('plot saved at %s/30_plots/%s' % (data_path, base_abb))
        elif save is True and base_abb is None:
            plt.savefig(data_path + '/30_plots/' + plot_name + suffix + '.png')
            plt.savefig(data_path + '/30_plots/' + plot_name + suffix + '.pdf')
            print('plot saved at %s/30_plots/' % data_path)
        else:
            plt.show(bbox_inches='tight')


def plot_swediff_boxplot(dest_path, diffs, y_lim=(-200, 600), save=[False, True], base_abb=['LB', 'JB']):
    """ Plot boxplot of differences of SWE from manual/laser/emlid data to Leica data
    """
    plt.close()
    diffs.dswe_manual.describe()
    diffs.dswe_laser.describe()
    diffs.dswe_emlid.describe()
    diffs[['dswe_manual', 'dswe_laser', 'dswe_emlid']].plot.box(ylim=y_lim, figsize=(3, 4.5), fontsize=12, rot=15)
    plt.grid()
    plt.ylabel('ΔSWE (mm w.e.)', fontsize=12)
    if save is True:
        plt.savefig(dest_path + '30_plots/' + base_abb + '/box_diffSWE.png', bbox_inches='tight')
        plt.savefig(dest_path + '30_plots/' + base_abb + '/box_diffSWE.pdf', bbox_inches='tight')
    else:
        plt.show()


def plot_solquality(data_path, amb_leica, amb_emlid, create_plot=[False, True], save=[False, True], suffix='', y_lim=(0, 100),
                    x_lim=(dt.date(2021, 11, 26), dt.date(2022, 12, 1))):
    """
    Plot quality of ambiguity resolution (1=fix, 2=float, 5=standalone) for high-end and low-cost rovers
    """
    if create_plot is True:
        plt.close()
        plt.figure()

        # calculate number of fixed and float ambiguity solutions per day
        nr_float_leica = amb_leica[(amb_leica == 2)].resample('D').count()
        nr_fixed_leica = amb_leica[(amb_leica == 1)].resample('D').count()
        nr_total_leica = nr_fixed_leica + nr_float_leica
        nr_amb_leica = pd.concat([nr_fixed_leica, nr_float_leica, nr_total_leica], axis=1).fillna(0).astype(int)
        nr_amb_leica.columns = ['Fixed', 'Float', 'Total']
        nr_float_emlid = amb_emlid[(amb_emlid == 2)].resample('D').count()
        nr_fixed_emlid = amb_emlid[(amb_emlid == 1)].resample('D').count()
        nr_total_emlid = nr_fixed_emlid + nr_float_emlid
        nr_amb_emlid = pd.concat([nr_fixed_emlid, nr_float_emlid, nr_total_emlid], axis=1).fillna(0).astype(int)
        nr_amb_emlid.columns = ['Fixed', 'Float', 'Total']
        # x day mean
        leica_fixed_mean = nr_fixed_leica.resample('5D').mean().interpolate()
        emlid_fixed_mean = nr_fixed_emlid.resample('5D').mean().interpolate()
        leica_fixed_mean = leica_fixed_mean.rename('5 day mean')
        emlid_fixed_mean = emlid_fixed_mean.rename('5 day mean')

        # plot number of fixed and float ambiguity solutions per day
        params = {'legend.fontsize': 'xx-large',
                  'axes.labelsize': 'xx-large',
                  'axes.titlesize': 'xx-large',
                  'xtick.labelsize': 'xx-large',
                  'ytick.labelsize': 'xx-large',
                  'legend.frameon': False}
        pylab.rcParams.update(params)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 9))

        nr_amb_leica[(nr_amb_leica != 0)].plot(ax=axes[0], linestyle='', marker='o', markersize=2, ylim=y_lim).grid()
        nr_amb_emlid[(nr_amb_emlid != 0)].plot(ax=axes[1], linestyle='', marker='o', markersize=2, ylim=y_lim).grid()
        #leica_fixed_mean.plot(ax=axes[0], linestyle='-', linewidth=2.2, color='#1f77b4', alpha=0.5, ylim=y_lim).grid()
        #emlid_fixed_mean.plot(ax=axes[1], linestyle='-', linewidth=2.2, color='#1f77b4', alpha=0.5, ylim=y_lim).grid()
        #axes[0].axhline(y=12.5, color='red', linestyle='-', alpha=0.15, linewidth=11)
        #axes[1].axhline(y=12.5, color='red', linestyle='-', alpha=0.15, linewidth=11)
        axes[0].axhline(y=5, color='red', linestyle='-', alpha=0.25, linewidth=22)
        axes[1].axhline(y=5, color='red', linestyle='-', alpha=0.25, linewidth=22)
        axes[0].axhline(y=50, color='grey', linestyle='--')
        axes[1].axhline(y=50, color='grey', linestyle='--')
        axes[0].yaxis.set_major_locator(MultipleLocator(10))
        axes[1].yaxis.set_major_locator(MultipleLocator(10))
        axes[0].xaxis.set_major_locator(MonthLocator())
        axes[1].xaxis.set_major_locator(MonthLocator())
        leg = axes[1].legend(loc='upper right')
        for line in leg.get_lines():
            line.set_linewidth(2.5)
            line.set_markersize(8)
        axes[0].set_xlim(x_lim)
        axes[1].set_xlim(x_lim)
        axes[0].set_ylabel('High-end solution')
        axes[1].set_ylabel('Low-cost solution')
        axes[0].get_legend().remove()

        if save is True:
            plt.savefig(
                data_path + '/30_plots/0b_Ambstate_' + str(x_lim[0].year) + '_' + str(x_lim[1].year)[-2:] + suffix + '.png',
                bbox_inches='tight')
            plt.savefig(
                data_path + '/30_plots/0b_Ambstate_' + str(x_lim[0].year) + '_' + str(x_lim[1].year)[-2:] + suffix + '.pdf',
                bbox_inches='tight')
        else:
            plt.show()
    else:
        pass





""" GNSS Reflectometry functions """


def prepare_orbits(sp3_outdir, raporbit_path, gnssir_path):
    """ Download, unzip, rename & rapid move orbits (need to match the gnssrefl input format!)
        :param sp3_outdir: Temporary output directory to store & convert downloaded orbit files
        :param raporbit_path: GFZ data server from where GNSS rapid orbits are downloaded
        :param gnssir_path: output directory for gnssrefl input (yearly directories)

    """
    # Q: download rapid orbits
    sp3_tempdir = get_orbits(sp3_outdir, raporbit_path)

    # Q: unzip orbits
    unzip_orbits(sp3_tempdir)

    # Q: rename orbit files (need to match the gnssrefl input name format!)
    rename_orbits(sp3_tempdir, gnssir_path)


def get_orbits(sp3_outdir, raporbit_path):
    """ Download, uzip, rename rapid orbits from GFZ Data Server: 'ftp://isdcftp.gfz-potsdam.de/gnss/products/rapid/w????/*.SP3*'
        (???? = gpsweek, sample sp3 = 'GFZ0OPSRAP_20230930000_01D_05M_ORB.SP3.gz')
        example orbit file: 'GFZ0OPSRAP_20231190000_01D_05M_ORB.SP3.gz'
        :param sp3_outdir: Temporary output directory to store & convert downloaded orbit files
        :param raporbit_path: GFZ data server from where GNSS rapid orbits are downloaded
        :return: sp3_tempdir
        """

    # Q: download, unzip, rename rapid orbits (need to match the gnssrefl input format!)
    # create temporary preprocessing folder in temp orbit dir
    sp3_tempdir = sp3_outdir + 'preprocessing/'
    if not os.path.exists(sp3_tempdir):
        os.makedirs(sp3_tempdir, exist_ok=True)

    # get the newest year, doy from orbit file from temp orbit dir
    yeardoy_newest = sorted(glob.glob(sp3_outdir + '*.gz'), reverse=True)[0].split('_')[1]
    year_newest = int(yeardoy_newest[:4])
    doy_newest = int(yeardoy_newest[4:7])

    # convert to gpsweek and day of week (dow)
    gpsweek_newest, dow_newest = gnsscal.yrdoy2gpswd(year_newest, doy_newest)

    # convert today to gpsweek and day of week (dow)
    gpsweek_today, dow_today = gnsscal.date2gpswd(date.today())

    # define ftp subdirectories to download newly available orbit files
    gpsweek_list = list(range(gpsweek_newest, gpsweek_today + 1, 1))

    for gpswk in gpsweek_list:
        download_path = raporbit_path + 'w' + str(gpswk) + '/'
        # download all .SP3 rapid orbits from ftp server's subfolders
        subprocess.call('wget -r -np -nc -nH --cut-dirs=4 -A .SP3.gz ' + download_path + ' -P ' + sp3_tempdir)

    print(colored("\nGFZ rapid orbits downloaded to: %s" % sp3_outdir, 'blue'))

    return sp3_tempdir


def unzip_orbits(sp3_tempdir):
    """ Unzip all orbit files in the temporary orbit processing directory
        example from orbit file: 'GFZ0OPSRAP_20231190000_01D_05M_ORB.SP3.gz' to '_GFZ0OPSRAP_20231190000_01D_05M_ORB.SP3'
        :param sp3_tempdir: temporary orbit processing directory
        """
    # unzip all files
    subprocess.call(r'7z e -y ' + sp3_tempdir + ' -o' + sp3_tempdir)
    print(colored("\nGFZ rapid orbits unzipped", 'blue'))


def rename_orbits(sp3_tempdir, gnssir_path, sp3_outdir):
    """ Rename & move orbit files (need to match the gnssrefl input name format!)
        example from 'GFZ0OPSRAP_20231190000_01D_05M_ORB.SP3' to 'GFZ0MGXRAP_20231190000_01D_05M_ORB.SP3'
        :param sp3_tempdir: temporary orbit processing directory
        :param gnssir_path: output directory for gnssrefl input (yearly directories)
        :param sp3_outdir: temporary output directory to store & convert downloaded orbit files
    """
    # rename & move extracted orbits to match the gfzrnx input format
    for orbit_file in glob.glob(sp3_tempdir + '*.SP3'):
        # define input and output filename
        infile = os.path.basename(orbit_file)
        outfile = infile[1:5] + 'MGX' + infile[8:]
        print('\nrename orbit file from: ', infile, ' to: ', outfile)

        # rename the file
        os.rename(orbit_file, os.path.dirname(orbit_file) + '/' + outfile)

        # define dest dir
        year = outfile.split('_')[1][:4]
        dest_dir = gnssir_path + 'data/' + year + '/sp3/'

        # move file to data directory for gnssrefl if it does not already exist
        if not os.path.exists(dest_dir + outfile):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            shutil.move(os.path.dirname(orbit_file) + '/' + outfile, dest_dir)
            print("orbit file moved to yearly sp3 dir %s" % dest_dir)
        else:
            os.remove(os.path.dirname(orbit_file) + '/' + outfile)
            print("file in destination already exists, move aborted, file removed")
    print(colored("\nGFZ rapid orbits renamed and moved to yearly (e.g. 2021): %s" % dest_dir, 'blue'))

    # move zipped original orbit files (.gz) to parent dir
    for f in glob.iglob(sp3_tempdir + '*.gz', recursive=True):
        shutil.move(f, sp3_outdir)
    print("original zipped orbit files (.gz) moved to parent dir %s" % sp3_outdir)

    # remove temporary preprocessing directory
    shutil.rmtree(sp3_tempdir)


def prepare_obs(dest_path, rin_temp, base_name, gnssir_path):
    """ copy, rename, convert, move rinex files
        :param rin_temp: temporary rinex processing folder
        :param dest_path: data destination path for python processing
        :param base_name: prefix of base rinex observation files, e.g. station name ('NMLB')
        :param gnssir_path: gnssrefl processing folder
        :return: year_start, year_end, doy_start, doy_end
    """
    # Q: copy & rename base rinex observations to temporary directory
    copy_obs(dest_path, rin_temp, base_name)

    # Q: Convert rinex3 to rinex2 files and resample to 30s sampling rate (instead of 1Hz)
    year_start, year_end, doy_start, doy_end = conv_obs(rin_temp, gnssir_path, base_name)

    return year_start, year_end, doy_start, doy_end


def copy_obs(dest_path, rin_temp, base_name):
    """ copy & rename base rinex observations to temporary directory
        :param rin_temp: temporary rinex processing folder
        :param dest_path: data destination path for python processing
        :param base_name: prefix of base rinex observation files, e.g. station name ('NMLB')
    """
    for rinex_file in sorted(glob.glob(dest_path + '3387*0.*[olng]'), reverse=True):
        # copy base rinex obs [o] and nav [lng] files
        copy_file_no_overwrite(dest_path, rin_temp, os.path.basename(rinex_file))

        # rename base rinex files if not exist
        outfile = base_name.lower() + os.path.basename(rinex_file)[4:]
        if not os.path.exists(rin_temp + '/' + outfile):
            print('rename file')
            os.rename(rin_temp + os.path.basename(rinex_file), rin_temp + '/' + outfile)
        else:
            os.remove(rin_temp + os.path.basename(rinex_file))

    print(colored("\nRinex3 files copied and renamed to: %s" % rin_temp, 'blue'))


def conv_obs(rin_temp, gnssir_path, base_name):
    """ Convert rinex3 to rinex2 files and resample to 30s sampling rate (instead of 1Hz) & get
        start/end year, doy from newly downloaded files (to not process older files again)
        :param rin_temp: temporary rinex processing folder
        :param gnssir_path: gnssrefl processing folder
        :param base_name: prefix of base rinex observation files, e.g. station name ('NMLB')
        :return: year_start, year_end, doy_start, doy_end
    """
    doy = []
    for rinex_file in sorted(glob.glob(rin_temp + '*o'), reverse=True):
        year = '20' + os.path.basename(rinex_file)[-3:-1]
        doy_new = os.path.basename(rinex_file).split('.')[0][-4:-1]
        doy.append(doy_new)
        if not os.path.exists(
                gnssir_path + 'data/rinex/' + base_name.lower() + '/' + year + '/' + os.path.basename(rinex_file)):
            print(rinex_file)
            if not os.path.exists(gnssir_path + 'data/rinex/' + base_name.lower() + '/' + year + '/'):
                os.makedirs(gnssir_path + 'data/rinex/' + base_name.lower() + '/' + year + '/', exist_ok=True)
            subprocess.call(
                r'gfzrnx -finp ' + rinex_file + ' -vo 2 -smp 30 -fout ' + gnssir_path + 'data/rinex/' + base_name.lower() + '/' + year + '/::RX2::')

    print(colored(
        "\nRinex3 files converted to rinex2 and moved to yearly (e.g. 2021): %s" % gnssir_path + 'data/rinex/' + base_name.lower() + '/' + year + '/',
        'blue'))

    # return start and end year, doy for GNSS-IR processing
    year_start = year[-1]  # '2021'
    doy_start = doy[-1]  # '330'
    year_end = year[0]
    doy_end = doy[0]

    return year_start, year_end, doy_start, doy_end


def read_gnssir(dest_path, reflecto_sol_src_path, base_name, yy='21', copy=False, pickle='nmlb'):
    """ Plot GNSS interferometric reflectometry (GNSS-IR) accumulation results from the high-end base station
        :param dest_path: path to processing directory
        :param reflecto_sol_src_path: path to local directory where the GNSS-IR results are processed/stored (e.g.: '//wsl.localhost/Ubuntu/home/sladina/test/gnssrefl/data/')
        :param base_name: name of reflectometry solution file base_name, e.g.: 'nmlb'
        :copy: copy (True) or not (False) reflectometry solutions from ubuntu localhost to local folder
        :param pickle: name of pickle file (default = 'nmlb.pkl')
        :return df_rh
    """
    # create local directory for GNSS-IR observations
    loc_gnssir_dir = dest_path + '20_solutions/' + base_name + '/rh2-8m_ele5-30/'
    os.makedirs(loc_gnssir_dir, exist_ok=True)

    # Q: copy GNSS-IR solution files (*.txt) from the local directory if not already existing
    if copy is True:
        print(colored("\ncopy new reflectometry solution files", 'blue'))
        # get list of yearly directories
        for f in glob.glob(reflecto_sol_src_path + '2*'):
            year = os.path.basename(f)
            if int(year) >= int('20' + yy):
                # copy missing reflectometry solution files
                for f in glob.glob(reflecto_sol_src_path + year + '/results/' + base_name.lower() + '/rh2-8m_ele5-30/*.txt'):
                    file = os.path.basename(f)
                    # skip files of 2021 before 26th nov (no gps data before installation)
                    if not os.path.exists(loc_gnssir_dir + file):
                        # check if the name of the solution file begins with the year
                        if file[:4] == year:
                            print(file)
                        else:
                            # rename GNSS-IR solution files from doy.txt to year_nmlbdoy.txt
                            os.rename(f, reflecto_sol_src_path + year + '/results/' + base_name.lower() + '/' + year + '_' + base_name.lower() + file)
                            print("\nGNSS-IR solution files renamed from %s to %s" % (file, year + '_' + base_name.lower() + file))

                        # copy the files
                        shutil.copy2(f, loc_gnssir_dir)
                        print("file copied from %s to %s" % (f, loc_gnssir_dir))
                    else:
                        # print(colored("\nfile in destination already exists: %s, \ncopy aborted!!!" % dest_path, 'yellow'))
                        pass
            else:
                pass
        print(colored("\nnew reflectometry solution files copied", 'blue'))
    else:
        print("\nno files to copy")

    # Q: read all existing GNSS-IR observations from .pkl if already exists, else create empty dataframe
    loc_gnssir_dir = dest_path + '20_solutions/' + base_name + '/rh2-8m_ele5-30/'
    path_to_oldpickle = loc_gnssir_dir + pickle + '.pkl'
    if os.path.exists(path_to_oldpickle):
        print(
            colored('\nReading already existing reflectometry solutions from pickle: %s' % path_to_oldpickle, 'yellow'))
        df_rh = pd.read_pickle(path_to_oldpickle)
        old_idx = df_rh.index[-1].date().strftime("%Y%j")
        old_idx_year = int(old_idx[:4])
        old_idx_doy = int(old_idx[-3:])
    else:
        print(colored('\nNo existing reflectometry solutions pickle!', 'yellow'))
        df_rh = pd.DataFrame()
        old_idx_year = 2021
        old_idx_doy = 330

    # read all reflector height solution files in folder, parse mjd column to datetimeindex and add them to the dataframe
    print(colored('\nReading all new reflectometry solution files from: %s' % loc_gnssir_dir + '*.txt', 'blue'))
    for file in glob.iglob(loc_gnssir_dir + '*.txt', recursive=True):
        # read solution files newer than last entry in reflectometry solutions pickle, check year and doy
        if ((int(os.path.basename(file)[:4]) >= old_idx_year) & (int(os.path.basename(file)[-7:-4]) > old_idx_doy)) \
                or (int(os.path.basename(file)[:4]) > old_idx_year):
            print(file)

            # header: year, doy, RH (m), sat,UTCtime (hrs), Azim (deg), Amp (v/v), eminO (deg), emaxO (deg), NumbOf (values), freq,rise,EdotF (hrs), PkNoise, DelT (min), MJD, refr-appl (1=yes)
            rh = pd.read_csv(file, header=4, delimiter=' ', skipinitialspace=True, na_values=["NaN"],
                             names=['year', 'doy', 'RH', 'sat', 'UTCtime', 'Azim', 'Amp', 'eminO', 'emaxO', 'NumbOf',
                                    'freq', 'rise', 'EdotF', 'PkNoise', 'DelT', 'MJD', 'refr-appl'], index_col=False)
            df_rh = pd.concat([df_rh, rh], axis=0)
        else:
            pass

    # convert year doy UTCtime to datetimeindex
    df_rh.index = pd.DatetimeIndex(pd.to_datetime(df_rh.year * 1000 + df_rh.doy, format='%Y%j')
                                   + pd.to_timedelta(df_rh.UTCtime, unit='h')).floor('s')

    # detect all dublicates and only keep last dublicated entries
    df_rh = df_rh[~df_rh.index.duplicated(keep='last')]

    # store dataframe as binary pickle format
    df_rh.to_pickle(loc_gnssir_dir + pickle + '.pkl')
    print(colored(
        '\nstored all old and new reflectometry solution data (without dublicates) in pickle: %s' + loc_gnssir_dir + pickle + '.pkl',
        'blue'))

    return df_rh


def filter_gnssir(df_rh, acc_at_time_of_first_obs=0, freq=[1, 5, 101, 102, 201, 202, 207, 'all', '1st', '2nd'], threshold=2, date_of_min_swe=None):
    """ Plot GNSS interferometric reflectometry (GNSS-IR) accumulation results from the high-end base station
        best results: ele=5-30, f=2, azi=30-160 & 210-310
        :param df_rh: GNSS-IR data
        :param acc_at_time_of_first_obs: The value of snow height above antenna at the time of first GNSS-IR observation. Default is 0, for the case of no snow above antenna by the start of observation
        :param freq: chosen satellite system frequency to use results, 1=gps l1, 5=gps l5, 101=glonass l1, 102=glonass l2, 201=galileo l1, 202=galileo l2, 207=galileo l5
        :param threshold: threshold for outlier detection (x sigma)
        :return df_rh, gnssir_acc, gnssir_acc_sel
    """

    # Q: select frequencies to analyze
    if freq == 'all':  # select all frequencies from all systems
        print('all frequencies are selected')
        gnssir_rh = df_rh[['RH', 'Azim']]
    elif freq == '2nd':  # select all second frequencies from GPS, GLONASS, GALILEO
        print('2nd frequencies are selected')
        gnssir_rh = df_rh[['RH', 'Azim']][(df_rh.freq.isin([5, 102, 205, 207]))]
    elif freq == '1st':  # select all first frequencies from GPS, GLONASS, GALILEO
        print('1st frequencies are selected')
        gnssir_rh = df_rh[['RH', 'Azim']][(df_rh.freq.isin([1, 101, 201]))]
    else:  # select chosen single frequency
        print('single frequency is selected')
        gnssir_rh = df_rh[['RH', 'Azim']][(df_rh.freq == freq)]

    # Q: excluding spuso e-m-wave bending and reflection zone azimuths and convert to mm
    gnssir_rh = gnssir_rh[(gnssir_rh.Azim > 30) & (gnssir_rh.Azim < 310)]
    gnssir_rh = gnssir_rh.RH[(gnssir_rh.Azim > 210) | (gnssir_rh.Azim < 160)] * 1000

    # Q: adjust for snow mast heightening (approx. 3m elevated several times a year)
    print('\ndata is corrected for snow mast heightening events (remove sudden jumps > 1m)')
    # sort index
    gnssir_rh = gnssir_rh.sort_index()

    # detect jump
    jump = gnssir_rh[(gnssir_rh.diff() > 2500)]  # detect jumps (> 2500mm) in the dataset

    # get value of jump difference (of values directly after - before jump)
    jump_ind = jump.index.format()[0]
    jump_val = gnssir_rh[jump_ind] - gnssir_rh[:jump_ind][-2]

    # detect and correct all jumps
    while jump.empty is False:
        print('\njump of height %s is detected! at %s' % (jump_val, jump.index.format()[0]))
        adj = gnssir_rh[
                  (gnssir_rh.index >= jump.index.format()[0])] - jump_val  # correct all observations after jump [0]
        gnssir_rh = pd.concat([gnssir_rh[~(gnssir_rh.index >= jump.index.format()[0])],
                               adj])  # concatenate all original obs before jump with adjusted values after jump
        jump = gnssir_rh[(gnssir_rh.diff() > 2500)]

    print('\nno jump detected!')

    # Q: remove outliers based on x*sigma threshold
    print('\nremove outliers based on %s * sigma threshold' % threshold)
    upper_limit = gnssir_rh.rolling('3D').median() + threshold * gnssir_rh.rolling('3D').std()
    lower_limit = gnssir_rh.rolling('3D').median() - threshold * gnssir_rh.rolling('3D').std()
    gnssir_rh_clean = gnssir_rh[(gnssir_rh > lower_limit) & (gnssir_rh < upper_limit)]

    # resample to 15min
    gnssir_rh_clean = gnssir_rh_clean.resample('15min').median().dropna()
    gnssir_rh_clean_daily = gnssir_rh_clean.resample('D').median()

    # Q: convert the distance between snow surface and antenna to accumulation (by subtracting all values from first observation)
    gnssir_acc = (gnssir_rh_clean_daily[0] - gnssir_rh_clean)
    # Q: adjust the first observation value to the height of snow above antenna at the time of first observation (by adding the accumulated snow measured by laser)
    gnssir_acc = gnssir_acc + acc_at_time_of_first_obs

    # resample accumulation data
    gnssir_acc_daily = gnssir_acc.resample('D').median()
    gnssir_acc_daily_std = gnssir_acc.resample('D').std()
    std_mean = gnssir_acc_daily_std.mean()
    std_percentual = gnssir_acc_daily_std * 100 / gnssir_acc_daily
    std_percentual_mean = std_percentual.mean()
    print('\nThe daily noise of GNSS-IR solution (mean of standard deviation of 15min solution from daily mean):'
          '%s kg/m2 (%s %%)' % (std_mean, std_percentual_mean))

    return gnssir_acc, gnssir_acc_daily, gnssir_acc_daily_std, gnssir_rh_clean
