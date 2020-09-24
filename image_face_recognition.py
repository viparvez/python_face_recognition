# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 02:55:04 2020

@author: Parvez
"""
#importing the required libraries

import cv2
import face_recognition

#load the sample images and get the 128 face embedings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

#load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file('images/testing/trump-modi.jpg')

#Save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encoding = [modi_face_encoding, trump_face_encoding]
known_face_names = ["Narendar Modi", "Donald Trump"]

#detect number of images
all_face_locations = face_recognition.face_locations(image_to_recognize,number_of_times_to_upsample=2,model='hog')
#detect_face_encodings for all images
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)



#image_to_read = cv2.imread('images/testing/trump-modi.jpg')
#cv2.imshow("test",image_to_read)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#detect number of images
#all_face_locations = face_recognition.face_locations(image_to_read,number_of_times_to_upsample=2,model='hog')

#print the number of faces
#print('There are {} no of faces in this image' .format(len(all_face_locations)))

#looping through the face locations
for index, current_face_location, current_face_encoding in zip(all_face_locations,all_face_encodings):
    #splitting the tupple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {} at top: {}, right: {}, bottom: {}, left: {}'.format(len(all_face_locations)))
    
    all_matches = face_recognition.compare_faces(all_face_encodings, current_face_encoding)
    name_of_person = 'Unknown face'
    
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
    cv2.rectangle(image_to_recognize,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    
    #Display the name
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_recognize, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
    cv2.imshow("Faces identified ",image_to_recognize)
    cv2.waitKey(0)