import ast
import re
import numpy as np
import pandas as pd
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from lib import aprioriFinal
from lib import knnFinal
from lib import kMeansFinal

def data_clean__age(frame):
    age_frame = pd.DataFrame()
    # -- Exclude Lowest && Highest k% percentage in order to remove outliers -- #
    lPosAge = int(0.02 * len(frame))
    hPosAge = int(0.98 * len(frame))
    # -- Assign age_frame -- #
    age_frame['age'] = frame.sort_values('Age')['Age'].iloc[lPosAge:hPosAge]
    return age_frame['age']

def data_clean__height(frame):
    height_frame = pd.DataFrame()
    # -- Exclude Lowest && Highest k% percentage in order to remove outliers -- #
    lPosHeight = int(0.02 * len(frame))
    hPosHeight = int(0.98 * len(frame))
    # -- Assign height_frame -- #
    height_frame['height'] = frame.sort_values('Height')['Height'].iloc[lPosHeight:hPosHeight].str.replace(r'\D', '').astype(int)
    return height_frame['height']

def data_clean__shoe_size(frame):
    shoe_size_frame = pd.DataFrame()
    # -- Exclude Lowest && Highest k% percentage in order to remove outliers -- #
    hPosShoeSize = int(0.98 * len(frame))
    # -- Assign shoe_size_frame -- #
    shoe_size_frame['shoe_size'] = frame.sort_values('Shoe Size')['Shoe Size'].iloc[0:hPosShoeSize].str.replace(',', '.').astype(float)
    # -- Only return values bigger than 17.0, which is the minimum expected one, according to reality -- #
    return shoe_size_frame[shoe_size_frame['shoe_size'] >= 17.0]

def data_clean__gender(frame):
    gender_frame = pd.DataFrame()
    # -- Assign gender_frame -- #
    gender_frame['gender'] = frame['Gender'].str[0].str.lower()
    gender_accepted_vals = ['f', 'm']
    # -- Turn everything into either 'f' or 'm' -- #
    return gender_frame[gender_frame['gender'].isin(gender_accepted_vals)]

def reduce_study_program_name(x):
    # -- Combine all SDT types into one SDT group -- #
    if "SDT" in x:
        return "SDT"
    if "GAMES" in x:
        return "GAMES"
    if "Guest" in x:
        return "GUEST"
    return x

def data_clean__study_program(frame):
    study_program_frame = pd.DataFrame()
    # -- Assign study_program_frame -- #
    study_program_frame['study_program'] = frame['What degree are you studying?'].str.rstrip()
    reduced_study_program_frame = study_program_frame['study_program'].map(lambda x: '{}'.format(reduce_study_program_name(x)),
                na_action=None)
    return reduced_study_program_frame

def data_clean__phone_os(frame):
    phone_os_frame = pd.DataFrame()
    # -- Assign phone_os_frame -- #
    phone_os_frame['phone_os'] = frame['Which phone OS do you prefer?'].str.rstrip()
    return phone_os_frame['phone_os']

def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)

def transform_prog_lang(x):
    # -- Sorting is important for feeding the input to Apriori, for instance -- #
    return tuple(sorted(map(str.strip, re.findall(r"[\w\#?\+?']+", x))))

def data_clean__prog_lang(frame):
    prog_lang_frame = pd.DataFrame()
    # -- Assign prog_lang_frame -- #
    prog_lang_frame['prog_lang'] = frame['Which programming languages do you know?'].str.lower()
    prog_lang_frame['prog_lang'] = prog_lang_frame['prog_lang'].apply(lambda x: transform_prog_lang(x))
    return prog_lang_frame['prog_lang']

def data_clean__pick_number(frame):
    pick_number_frame = pd.DataFrame()
    # -- Assign pick_number_frame -- #
    pick_number_frame['pick_number'] = frame['Pick a number']
    transform = pick_number_frame[pick_number_frame['pick_number'] != 'Asparagus']
    return transform

def map_to_int(x):
    return [int(math.floor(float(i))) for i in x]

def data_clean__random_numbers(frame):
    random_numbers_frame = pd.DataFrame()
    # -- Assign random_numbers_frame -- #
    random_numbers_frame['random_numbers'] = frame['Write four (4) random numbers between 0 and 15']
    # -- Eliminate answers that do not include 4 numbers, as requested by the question -- #
    random_numbers_frame['random_numbers'] = random_numbers_frame['random_numbers'].str.split(',').apply(lambda x: map_to_int(x) if len(x) == 4 else 'false')
    transform = random_numbers_frame[random_numbers_frame['random_numbers'] != 'false']
    return transform

def model__prog_lang_number(frame):
    prog_lang_number_frame = pd.DataFrame()
    prog_lang_number_frame['prog_lang_number'] = frame.apply(lambda x: len(x)*1.0)
    return prog_lang_number_frame['prog_lang_number']

def model_course_reason(long_course_reason):
    # -- Do data transformation on the course reason options, in order to ensure easiness before feeding the input to the algorithms -- #
    if long_course_reason == 'I am interested in the subject':
        return 'i'
    if long_course_reason == 'It may help me to find a job':
        return 'j'
    if long_course_reason == 'This was a mandatory course for me':
        return 'm'
    if long_course_reason == 'The other optional courses were least appealing':
        return 'a'
    return 'o'

def data_clean__course_reason(frame):
    course_reason_frame = pd.DataFrame()
    # -- Assign course_reason_frame -- #
    course_reason_frame['course_reason'] = frame['Why are you taking this course?'].apply(lambda x: model_course_reason(x))
    return course_reason_frame['course_reason']

def data_clean__games_number(frame):
    games_number_frame = pd.DataFrame()
    games_number_frame['games_number'] = frame['Which of these games have you played?'].str.split(';').apply(lambda x: len(x)*1.0)
    return games_number_frame['games_number']

def merge_cols_for_clustering_with_label(label_to_cluster_on, age_frame, shoe_size_frame, height_frame, study_program_frame, gender_frame):
    if label_to_cluster_on == 'study_program':
        return pd.concat([age_frame, shoe_size_frame, height_frame, study_program_frame], axis=1)
    return pd.concat([age_frame, shoe_size_frame, height_frame, gender_frame], axis=1)


def clean_data_attributes(data):
    # -- Clean age -- #
    frame_age_data = data
    age_frame = data_clean__age(frame_age_data)
    age_frame__no_duplicates_set = age_frame.drop_duplicates().values.astype(int).tolist()

    # -- Clean height -- #
    frame_height_data = data
    height_frame = data_clean__height(frame_height_data)
    height_frame__no_duplicates_set = height_frame.drop_duplicates().values.astype(int).tolist()

    # -- Clean shoe size -- #
    frame_shoe_size_data = data
    shoe_size_frame = data_clean__shoe_size(frame_shoe_size_data)
    shoe_size_frame__no_duplicates_set = shoe_size_frame.drop_duplicates().values.astype(int).tolist()

    # -- Clean gender -- #
    frame_gender_data = data
    gender_frame = data_clean__gender(frame_gender_data)
    gender_frame__no_duplicates_set = gender_frame.drop_duplicates().values.tolist()

    # -- Clean study program -- #
    frame_study_program_data = data
    study_program_frame = data_clean__study_program(frame_study_program_data)
    study_program_frame__no_duplicates_set = study_program_frame.drop_duplicates().values.tolist()

    # -- Clean phone_os program -- #
    frame_phone_os_data = data
    phone_os_frame = data_clean__phone_os(frame_phone_os_data)
    phone_os_frame__no_duplicates_set = phone_os_frame.drop_duplicates().values.tolist()

    # -- Clean prog_lang -- #
    frame_prog_lang_data = data
    prog_lang_frame = data_clean__prog_lang(frame_prog_lang_data)
    prog_lang_frame__no_duplicates_set = prog_lang_frame.drop_duplicates().values.tolist()

    # -- Model how many prog_lang each student knows -- #
    prog_lang_number_frame = model__prog_lang_number(prog_lang_frame)
    prog_lang_number_frame__no_duplicates_set = map(int, prog_lang_number_frame.drop_duplicates().values.tolist())

    # -- Model how many games each student played -- #
    frame_games_number_data = data
    games_number_frame = data_clean__games_number(frame_games_number_data)

    # -- Clean pick_number -- #
    frame_pick_number_data = data
    pick_number_frame = data_clean__pick_number(frame_pick_number_data)
    pick_number_frame__no_duplicates_set = pick_number_frame.drop_duplicates().values.tolist()

    # -- Clean pick_random_numbers -- #
    frame_random_numbers_data = data
    random_numbers_frame = data_clean__random_numbers(frame_random_numbers_data)

    # -- Clean course_reason -- #
    frame_course_reason_data = data
    course_reason_frame = data_clean__course_reason(frame_course_reason_data)
    course_reason_frame__no_duplicates_set = course_reason_frame.drop_duplicates().values.tolist()

    return age_frame, height_frame, shoe_size_frame, gender_frame, study_program_frame, phone_os_frame, prog_lang_frame, prog_lang_number_frame, games_number_frame, pick_number_frame, random_numbers_frame, course_reason_frame


def normalize_col(col):
    col = [float(i) for i in col]
    # -- Normalize data column -- #
    min_val = min(col)
    max_val = max(col)
    for i, val in enumerate(col):
        col[i] = (val - min_val) / (max_val - min_val)
    return col


def run_pattern_mining(prog_lang_frame, random_numbers_frame):
    print "#----------------------------------- START: Pattern Mining (with Apriori) ---------------------------------------#"
    # 1. Find patterns on => numbers picked in the range [0, 15] #
    print "1. Find patterns on => numbers picked in the range [0, 15]"
    apriori_data_list_1 = []
    for tuple in random_numbers_frame['random_numbers']:
        apriori_data_list_1.append(tuple)
    aprioriFinal.main(apriori_data_list_1, 6, 0.3)

    print

    # 2. Find patterns on => prog_lang #
    print "2. Find patterns on => prog_lang"
    apriori_data_list_2 = []
    for tuple in prog_lang_frame:
        apriori_data_list_2.append(list(tuple))
        aprioriFinal.main(apriori_data_list_2, 10, 0.5)
    print "#----------------------------------- END: Pattern Mining (with Apriori) ---------------------------------------#"
    print


def map_to_categories_prog_lang_number_frame(prog_lang_number_frame):
    for i, prog_lang_num in enumerate(prog_lang_number_frame):
        if prog_lang_num < 4.0:
            prog_lang_number_frame[i] = 'l'
        if prog_lang_num > 7.0:
            prog_lang_number_frame[i] = 'h'
        if prog_lang_num <= 7.0 and prog_lang_num >= 4.0:
            prog_lang_number_frame[i] = 'm'
    return prog_lang_number_frame



def run_classification(study_program_frame, prog_lang_number_frame, course_reason_frame, gender_frame, pick_number_frame):
    print "#----------------------------------- START: Classification (with k-Nearest Neighbours) ---------------------------------------#"
    # mapped_to_categories_prog_lang_number_frame = map_to_categories_prog_lang_number_frame(prog_lang_number_frame)
    normalized_prog_lang_number_frame = pd.DataFrame({"prog_lang_number": normalize_col(prog_lang_number_frame)})
    normalized_picked_number_frame = pd.DataFrame({"pick_number": normalize_col(pick_number_frame['pick_number'])})

    knn_dataset = pd.concat(
        [study_program_frame, normalized_prog_lang_number_frame, course_reason_frame, gender_frame, normalized_picked_number_frame], axis=1)
    knn_dataset__non_nan_vals = knn_dataset[knn_dataset['gender'].notnull()]
    knn_dataset__non_nan_vals = knn_dataset__non_nan_vals[knn_dataset__non_nan_vals['pick_number'].notnull()]
    knnFinal.main(knn_dataset__non_nan_vals)
    print "#----------------------------------- END: Classification (with k-Nearest Neighbours) -----------------------------------------#"
    print


def plot__shoe_size__vs__height(plots):
    fig = plt.figure()
    plt.plot(plots[0][0]['f'], plots[0][1]['f'], 'rX',
             plots[0][0]['m'], plots[0][1]['m'], 'rH',
             plots[1][0]['f'], plots[1][1]['f'], 'gX',
             plots[1][0]['m'], plots[1][1]['m'], 'gH')
    plt.ylabel('shoe size')
    plt.xlabel('height')
    plt.show()


def plot__shoe_size__vs__height__vs__age__3d(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, cluster in enumerate(clusters):
        for point in clusters[i]:
            xs = point[0]
            ys = point[1]
            zs = point[2]
            label_index = len(point) - 1
            if i == 0:
                if point[label_index] == 'f':
                    m = 'h'
                    c = 'turquoise'
                if point[label_index] == 'm':
                    m = 'H'
                    c = 'darkturquoise'
            if i == 1:
                if point[label_index] == 'f':
                    m = 'x'
                    c = 'orange'
                if point[label_index] == 'm':
                    m = 'X'
                    c = 'orangered'
            ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.set_xlabel('Age')
    ax.set_ylabel('Shoe Size')
    ax.set_zlabel('Height')
    plt.show()



def run_clustering(age_frame, shoe_size_frame, height_frame, study_program_frame, gender_frame):
    print "#----------------------------------- START: Clustering (with k-Means) -----------------------------------------#"
    label_to_cluster_on = 'gender'
    kMeans_dataset = merge_cols_for_clustering_with_label(label_to_cluster_on, age_frame, shoe_size_frame, height_frame, study_program_frame, gender_frame)
    kMeans_dataset__non_nan_vals_for_age = kMeans_dataset[kMeans_dataset['age'].notnull()]
    kMeans_dataset__non_nan_vals_for_shoe_size = kMeans_dataset__non_nan_vals_for_age[kMeans_dataset__non_nan_vals_for_age['shoe_size'].notnull()]
    kMeans_dataset__non_nan_vals_for_height = kMeans_dataset__non_nan_vals_for_shoe_size[kMeans_dataset__non_nan_vals_for_shoe_size['height'].notnull()]
    kMeans_dataset__non_nan_vals = kMeans_dataset__non_nan_vals_for_height[kMeans_dataset__non_nan_vals_for_height[label_to_cluster_on].notnull()]

    # -- Eliminate more outliers where the proportion between shoe size and height is weird according to reality -- #
    kMeans_dataset__non_nan_vals = kMeans_dataset__non_nan_vals[kMeans_dataset__non_nan_vals['height'] / kMeans_dataset__non_nan_vals['shoe_size'] <= 6.5]

    # -- Run kMeans Algorithm and get the clusters -- #
    clusters = kMeansFinal.main(kMeans_dataset__non_nan_vals.as_matrix(), 2)

    # -- Prepare points for plotting 2d and 3d -- #
    plots = []
    for i, cluster in enumerate(clusters):
        coord_y = {'m': [], 'f': []}  # coord_y == shoe size
        coord_x = {'m': [], 'f': []}  # coord_x == height
        coord_z = {'m': [], 'f': []}  # coord_z == age
        for point in clusters[i]:
            if point[len(point)-1] == 'f':
                coord_y['f'].append(point[1])
                coord_x['f'].append(point[2])
                coord_z['f'].append(point[0])
            else:
                coord_y['m'].append(point[1])
                coord_x['m'].append(point[2])
                coord_z['m'].append(point[0])
        plots.append([coord_x, coord_y, coord_z])

    # -- Plot 2d && 3d -- #
    # plot__shoe_size__vs__height(plots)
    plot__shoe_size__vs__height__vs__age__3d(clusters)

    print "#----------------------------------- END: Clustering (with k-Means) -----------------------------------------#"


def load_dataframe_cols(data):
    # -- Clean attributes -- #
    age_frame, height_frame, shoe_size_frame, gender_frame, study_program_frame, phone_os_frame, prog_lang_frame, prog_lang_number_frame, games_number_frame, pick_number_frame, random_numbers_frame, course_reason_frame = clean_data_attributes(data)


    # -- Run PATTERN MINING (apriori) -- #
    # 1. Find patterns on => numbers picked in the range 0-15 #
    # 2. Find patterns on => known programming languages #
    run_pattern_mining(prog_lang_frame, random_numbers_frame)

    # -- Run CLASSIFICATION (knn) -- #
    # 1. Study Line == ? if knowing: #prog_lang_number, course reason, gender, number picked
    run_classification(study_program_frame, prog_lang_number_frame, course_reason_frame, gender_frame, pick_number_frame)


    # -- Run CLUSTERING (kMeans) -- #
    # 1. Age, Shoe Size, Height, Gender == Label
    run_clustering(age_frame, shoe_size_frame, height_frame, study_program_frame, gender_frame)


def main():
    data_file_name = 'Data Mining - Spring 2017.csv'

    # -- Read dataset --#
    data = pd.read_csv(data_file_name, sep=',') # .as_matrix()
    load_dataframe_cols(data)

main()

