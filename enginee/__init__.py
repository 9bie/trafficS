# coding:utf-8
from __future__ import print_function

import sys
import cv2
import numpy as np
class MutiTracker(object):
    """
    Modes of MutiTracker.
    Don't use two or more mod at the same time!!!

    @param videoPath: Support http(Expected) or rtmp(Expected) or local mp4 path.
    @param boundary_line: Four direction(up, down, left, right,number) boundary for recoding prople Count.
    @param boundary_line: If not point than will do not recoding this direction. ( This direction counter always zero. )
    @param boundary_line: If you not set number, it will use the default configuration.the number can set the line Whether close to the edge.
    """
    def __init__(self, videoPath,boundary_line=(False,False,False,False,10)):
        super(MutiTracker, self).__init__()
        self.videoPath = videoPath
        self.CV_CAP_PROP_FRAME_WIDTH = 3
        self.CV_CAP_PROP_FRAME_HEIGHT = 4
        self.trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.cap = cv2.VideoCapture(self.videoPath)
        self.w = self.cap.get(self.CV_CAP_PROP_FRAME_WIDTH)
        self.h = self.cap.get(self.CV_CAP_PROP_FRAME_HEIGHT)
        self.flags = boundary_line # w,s,a,d
        self.point_down,self.point_up,self.point_left,self.point_right = 0,0,0,0
        self.line_up,self.line_down,self.line_left,self.line_right = None,None,None,None
        self.occupy = False
        success, self.frame = self.cap.read()
        if not success:
            raise 'Failed to read video'
        if self.flags[0] == False and self.flags[1] == False and self.flags[2] == False and self.flags[3] == False:
            raise "No boundary set."
    
        if self.flags[0]:
            self.point_up = int(1*(self.h/self.flags[4]))# The line's height.
            self.line_up = np.array([[0,self.point_up],[self.w,self.point_up]],np.int32).reshape((-1,1,2))
            
        if self.flags[1]:
            self.point_down = int((self.flags[4]-1)*(self.h/self.flags[4]))
            self.line_down = np.array([[0,self.point_down],[self.w,self.point_down]],np.int32).reshape((-1,1,2))
            
        if self.flags[2]:
            self.point_left = int(1*(self.w/self.flags[4]))
            self.line_left = np.array([[self.point_left,0],[self.point_left,self.h]],np.int32).reshape((-1,1,2))
            
        if self.flags[3]:
            self.point_right = int((self.flags[4]-1)*(self.w/self.flags[4]))
            self.line_right = np.array([[self.point_right,0],[self.point_right,self.h]],np.int32).reshape((-1,1,2))
        
    def __draw_line(self):
        if self.flags[0]:
            self.frame = cv2.polylines(self.frame,[self.line_up],False,(255,255,255),thickness=1)
        if self.flags[1]:
            self.frame = cv2.polylines(self.frame,[self.line_down],False,(255,255,255),thickness=1)
        if self.flags[2]:
            self.frame = cv2.polylines(self.frame,[self.line_left],False,(255,255,255),thickness=1)
        if self.flags[3]:
            self.frame = cv2.polylines(self.frame,[self.line_right],False,(255,255,255),thickness=1)
    def __createTrackerByName(self,trackerType):
        # Create a tracker based on tracker name
        if trackerType == self.trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == self.trackerTypes[1]: 
            tracker = cv2.TrackerMIL_create()
        elif trackerType == self.trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == self.trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == self.trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == self.trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == self.trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == self.trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in self.trackerTypes:
                print(t)
        return tracker
    def Tracker1(self,frame_list=[],trackerType = "KCF", step=20):
        '''
        Reserved for Neural Networks's Interface.
        Using Neural Networks's processed people frame and opencv built-in mutitracker to tracking target and statistics.
        Of couse you can manual selector the frame.
        Suitable for outdoor cameras or big venue.

        @param frame_list: The enginee Processed the people's frame point tuple .if not frame then use Manual selector.
        @param trackerType： choose one 'BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'.
        @param step: Loop frequency. make sure tracker correct Identification target,we need Correction error.

        @return frame: now Monitor's image for CNN engine processing.
        @return u,d,l,r: if someone left the screen.the function will be return.
        '''
        Processed =[]
        up,down,left,right = 0,0,0,0
        manual = False
        if self.occupy:
            print("[warrning] Don't use two mode at same time.")
        self.occupy = True
        if  frame_list == []:
            manual = True
            self.__draw_line()
            while True:
                # Just use to test development environment.
                bbox = cv2.selectROI('MultiTracker', self.frame)
                frame_list.append(bbox)
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                k = cv2.waitKey(0) & 0xFF
                if (k == 113):  # q is pressed
                  break
        print('Selected bounding boxes {}'.format(frame_list))
        multiTracker = cv2.MultiTracker_create()
        for bbox in frame_list:
            multiTracker.add(self.__createTrackerByName(trackerType), self.frame, bbox)
        first = True
        for i in range(step):
            if not self.cap.isOpened():
                raise "Video Closed."
            
            success, self.frame = self.cap.read()
            if not success:
                print("Read Failed.Pass")
                break

            success, boxes = multiTracker.update(self.frame)
            for i, newbox in enumerate(boxes):
                # print(Processed)
                
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                p3 = (int(newbox[0]+(newbox[2])/2),int(newbox[1]+(newbox[3]/2)))# midpoint
                if self.flags[0] and i not in Processed:#w
                    if p3[1] <= self.point_up:
                        if not first:
                            up+=1
                        Processed.append(i)
                elif self.flags[1] and i not in Processed:#s
                    if p3[1] >= self.point_down:
                        if not first:
                            down+=1
                        Processed.append(i)
                elif self.flags[2] and i not in Processed:#a
                    if p3[0] <= self.point_left:
                        if not first:
                            left+=1
                        Processed.append(i)
                elif self.flags[3] and i not in Processed:#d
                    if p3[0] >= self.point_right:
                        if not first:
                            right+=1
                        Processed.append(i)
                if manual :
                    self.__draw_line()
                    if i not in Processed:
                        
                        cv2.rectangle(self.frame, p1, p2, (255,255,255), 2, 1)
                        cv2.circle(self.frame,p3, 3, (0,0,255), -1)
                        cv2.putText(self.frame, str(i),(p3),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1,cv2.LINE_AA)
            if manual:
                cv2.imshow('MultiTracker', self.frame)
            first = False
            if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                break
        self.occupy = False
        return self.frame,(up,down,left,right)
    def Tracker2(self,step=20):
        '''
        Based on Edge recognition's tracker.
        Very low recognition rate.but it very fast and have low occupancy.low
        Suitable for indoor cameras or small venue.

        Maybe will achieve hahaha.
        Or reference github:https://github.com/akko29/People-Counter
        '''
        if self.occupy:
            print("[warrning] Don't use two mode at same time.")
        self.occupy = True