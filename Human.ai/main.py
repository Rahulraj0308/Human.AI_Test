import cv2
import csv
import os
import pandas as pd
import mediapipe as mp
from GazeTracking import gaze_tracking
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def initialize_gaze_tracker():
    return gaze_tracking.GazeTracking()

def initialize_hand_tracker():
    mp_hands = mp.solutions.hands.Hands()
    return mp_hands

def process_video(video_file, gaze_tracker, hand_tracker):
    video = cv2.VideoCapture(video_file)
    gaze_positions_x = []
    gaze_positions_y = []
    hand_positions_x = []
    hand_positions_y = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gaze_tracker.refresh(frame)
        results = hand_tracker.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    hand_positions_x.append(landmark.x)
                    hand_positions_y.append(landmark.y)

        text = ""
        if gaze_tracker.is_right():
            text = "Looking right"
        elif gaze_tracker.is_left():
            text = "Looking left"
        elif gaze_tracker.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("Demo", frame)

        gaze_positions_x.append(gaze_tracker.horizontal_ratio())
        gaze_positions_y.append(gaze_tracker.vertical_ratio())

        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()

    return gaze_positions_x, gaze_positions_y, hand_positions_x, hand_positions_y

def plot_results_image(gaze_positions_x, gaze_positions_y, hand_positions_x, hand_positions_y):
    plt.figure(figsize=(10, 6))
    plt.plot(gaze_positions_x, label='Horizontal Gaze Position', color='blue')
    plt.plot(gaze_positions_y, label='Vertical Gaze Position', color='green')
    plt.plot(hand_positions_x, label='Horizontal Hand Position', linestyle='--', color='red')
    plt.plot(hand_positions_y, label='Vertical Hand Position', linestyle='--', color='orange')
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.title('Eye and Hand Movement Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_results_data(df):
    
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True,  fmt=".2f")
    plt.plot(df['MediaTime'], df['Velocity'])
    plt.plot(df['MediaTime'], df['Throttle'])
    plt.plot(df['MediaTime'], df['Brake'])
# Add title
    plt.title('Correlation Heatmap')
    plt.gcf().set_size_inches(20, 12)
    plt.show()
    pass
def modify_dataset(df):
    df["MediaTime"]=df["MediaTime"].round(1)*10
    df=df[["MediaTime","LonAccel","LatAccel","Throttle","Brake","Gear","Heading","HeadwayDistance","HeadwayTime","Lane","LaneOffset","RoadOffset","Steer","Velocity"]]
    df['LonAccel'] = df['LonAccel'].apply(lambda x: format(float(x), '.6f'))
    df['LatAccel'] = df['LatAccel'].apply(lambda x: format(float(x), '.6f'))
    df['Velocity'] = df['Velocity'].apply(lambda x: format(float(x), '.6f'))
def insert_image(image,df1,i):
    height, width, _ = image.shape
    part3 = image[height//2:, :width//2]


# Save the parts into separate folders
    cv2.imwrite('part_all/part'+str(i)+'.jpg', part3)
    
    #dicts = {'Name': 'Amy', 'Maths': 89, 'Science': 93} 
    #df = df.append(df2, ignore_index = True)
    df1.loc[len(df1.index)] = ['part_3/part'+str(i)+'.jpg'] 
    print("Parts saved successfully.")
def creat_dataframe():
    column_names=["imp"]
    df = pd.DataFrame({col_name: [] for col_name in column_names})
    return df
def get_images(video_file,MediaTime):
    vidcap=cv2.VideoCapture(video_file)
    count=0
    os.makedirs('part_3',exist_ok=True)
    df1=creat_dataframe()
    i=0
    for j in MediaTime:
        success,image=vidcap.read()
        if success!=True:
            break
        if count%30==0:
            insert_image(image,df1,i)
            i+=1
    
        count=j
def preprocess_image(img):
    try:
        img = cv2.resize(img, (100, 100))  
        img = img / 255.0  
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def load_classification_model(weights_path):
    try:
        model = load_model(weights_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_image_classes_from_df(df, column_name, model):
    
    predictions = []
    for img_path in df[column_name]:
        img = cv2.imread(img_path)
        img = preprocess_image(img)
        if img is not None:
            # Reshape image for prediction
            img = np.expand_dims(img, axis=0)
            # Make prediction
            prediction = model.predict(img)
            predictions.append(prediction.ravel())
    
    return np.array(predictions)

def main():
    
    video_file = input("Enter the path to the MP4 file: ")
    data_file = input("Enter the path to the .dat file: ")
    
    gaze_tracker = initialize_gaze_tracker()
    hand_tracker = initialize_hand_tracker()
    with open(data_file, 'r') as infile, open('output.csv', 'w') as outfile:
        for line in infile:
            outfile.write(line.replace(' ', ','))
    df=pd.read_csv("output.csv")
    
    
    gaze_positions_x, gaze_positions_y, hand_positions_x, hand_positions_y = process_video(video_file, gaze_tracker, hand_tracker)
    df=modify_dataset(df)
    MediaTime=df["MediaTime"]
    get_images(video_file,MediaTime)
    weights_path = "my_model.keras"  # Path to pre-trained model file
    model = load_classification_model(weights_path)
    if model:
        predictions = predict_image_classes_from_df(df, "imp", model)
    
        avg_probabilities = np.mean(predictions, axis=0)
    
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(avg_probabilities) + 1), avg_probabilities)
        plt.xlabel('Class Label')
        plt.ylabel('Average Probability')
        plt.title('Average Prediction Probabilities for All Images')
        plt.xticks(range(1, len(avg_probabilities) + 1))
        plt.grid(axis='y')
        plt.show()
    else:
        print("Failed to load the pre-trained model.")
    plot_results_image(gaze_positions_x, gaze_positions_y, hand_positions_x, hand_positions_y)
    plot_results_data(df)
    
if __name__ == "__main__":
    main()

