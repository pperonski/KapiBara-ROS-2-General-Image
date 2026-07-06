#!/usr/bin/env python3
from uuid import uuid4
import codon

import os
import rclpy
from rclpy.node import Node
import tf_transformations
from sklearn.neighbors import KDTree

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from sensor_msgs.msg import PointCloud2

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float64MultiArray

from kapibara.DeepIDTFLite import DeepIDTFLite

from kapibara_interfaces.msg import Emotions
from kapibara_interfaces.msg import Microphone
from kapibara_interfaces.msg import PiezoSense
from kapibara_interfaces.msg import PointValue2D

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

import webrtcvad

import cv2
from cv_bridge import CvBridge

import numpy as np

from timeit import default_timer as timer

import librosa



'''

A python package that is responsible for emotion asociative
memory and emotion estimation from sensor data.
It will publish value points that will be used by
actractor node to move towards point with most positive
value.

'''


ID_TO_EMOTION_NAME = [
    "angry",
    "fear",
    "happiness",
    "uncertainty",
    "boredom"
    ]

IMG_EMBEDDING_SHAPE = 32*32*4
ML_SPECTOGRAM_EMBEDDING_SHAPE = 32*32

# month, day, hour, minute, second
TIME_EMBEDDING_SHAPE = 5

OBJECT_DB = "object_database"
OBJECT_EMBEDDING_SHAPE = 32*32

POINT_MAP_DB = "points_database"
POINT_IMG_DB = "points_imgages"
POINT_SPEC_DB = "points_spect"
POINT_TIME_DB = "points_time"

FACE_TOLERANCE = 0.94

import chromadb

class DetectionPipeline:
    def __init__(self,cls_model_path:str,detection_model_path:str):
        self.cls_model = YOLO(cls_model_path)
        self.detection_model = YOLO(detection_model_path)
        
    def run(self,frame):
        """
        A function that performs detection and classification
        pipeline.
        
        It returns list with results, each containing:
            - embedding
            - x postion
            - y position
            - width
            - height
            - confidence score
        """        
        output = []
        
        results = self.detection_model(frame)
        
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                
                obj = frame[y1:y2,x1:x2]
                
                cls_result = self.cls_model(obj)
                                
                for res in cls_result:
                    output.append(
                        (
                            res.probs.data.cpu().numpy(),
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1,
                            conf
                        )
                    )
                    
        return output
    
    def get_class_names(self):
        return self.cls_model.names

class MapMind(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('tick_time', 0.05)
        
        package_path = get_package_share_directory('kapibara')
        
        self.declare_parameter('yolo_model_cls_path','yolov8n-cls_full_integer_quant.tflite')
        self.declare_parameter('yolo_model_path','yolov8n_full_integer_quant.tflite')
        self.declare_parameter('deepid_model_path','deepid.tflite')
        
        self.max_linear_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        yolo_model_cls_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
                
        self.detection_pipeline = DetectionPipeline(yolo_model_cls_path,yolo_model_path)
        
        deepid_model_path = self.get_parameter('deepid_model_path').get_parameter_value().string_value
        
        self.deep_id = DeepIDTFLite(filepath=os.path.join(package_path,'model',deepid_model_path))
        
        self.twist_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.emotion_pub = self.create_publisher(Emotions,'emotions', 10)
        
        self.spectogram_publisher = self.create_publisher(Image, 'spectogram', 10)
        
        self.value_point_pub = self.create_publisher(PointValue2D,'val_points',10)
        
        self.timer_period = self.get_parameter('tick_time').get_parameter_value().double_value  # seconds
        self.timer = self.create_timer(self.timer_period, self.tick_callback)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        self.image_sub = self.create_subscription(
            Image,
            'camera',
            self.image_callback,
            10)
        
        self.image_sub = self.create_subscription(
            Image,
            'depth',
            self.depth_image_callback,
            10)
        
        self.points_sub = self.create_subscription(
            PointCloud2,
            'points',
            self.points_callback,
            10)
        
        self.ext_emotion_sub = self.create_subscription(
            Emotions,
            'ext_emotion',
            self.ext_emotion_callback,
            10)
        
        self.mic_subscripe = self.create_subscription(Microphone,'mic',self.mic_callback,10)
                
        self.bridge = CvBridge()
                
        self.initialize_db()
        
        # A list of objects found in the scene, 64x64 embeddings with bouding boxes
        self.found_objects = []
        
        # A map of points with X,Y positions and associated emotional state, and
        # decay factor, we can use use color as a point age
        self.point_map = []
        self.point_map_scores = []
                
        self.pat_detected = 0.0
                
        self.emotion_state = 0.0
                
        self.image = None
        self.depth = None
        self.spectogram = None
                
        self.position = np.zeros(3)
        
        self.emotion = Emotions()
        
        # emotion state from external sources
        self.ext_emotion = Emotions()
        
        self.obstacle_detected = False
        
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        
        self.audio_fear = 0
        
        self.mic_buffor = np.zeros(2*16000,dtype=np.float32)
        
        self.wait_for_img = True
        self.wait_for_depth = True
        
        self.last_score = 0
        
        # robot position in the map
        self.x = 0
        self.y = 0
        self.yaw = 0
        
        self.last_points = None
        self.points_shape = None
        
        # angry
        # fear
        # happiness
        # uncertainty
        # boredom 
        # anguler values for each emotions state
        self.emotions_angle=[0.0,180.0,25.0,145.0,90.0] 
        
        self.ears_publisher = self.create_publisher(
            Float64MultiArray,
            'ears_controller/commands', 
            10
            )
        
    def reinitialize_db(self):
        
        self.initialize_db()
        
    def initialize_db(self):
        
        self.associated_db = chromadb.PersistentClient("association_db")
        self.obj_db = self.associated_db.create_collection("obj",get_or_create=True)
        
        self.point_map_db = self.associated_db.create_collection("cloud",get_or_create=True)
        
    def odom_callback(self,odom:Odometry):
        self.position = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
            ])
        
        self.yaw = tf_transformations.euler_from_quaternion(odom.pose.pose.orientation)[2]
    
    def mic_callback(self,mic:Microphone):
        
        self.get_logger().debug("Mic callback")
        
        self.mic_buffor = np.roll(self.mic_buffor,mic.buffor_size)
        
        left = np.array(mic.channel1,dtype=np.float32)/np.iinfo(np.int32).max
        right = np.array(mic.channel2,dtype=np.float32)/np.iinfo(np.int32).max
        
        combine = ( left + right ) / 2.0
        
        self.mic_buffor[:mic.buffor_size] = combine[:]
        
        start = timer()
        
        spectogram = librosa.feature.melspectrogram(y=self.mic_buffor, sr=16000,n_mels=224,hop_length=143)
        
        self.get_logger().info(f"Spectogram size: {spectogram.shape}")
        
        # publish last spectogram
        self.spectogram_publisher.publish(self.bridge.cv2_to_imgmsg(spectogram))
                
        self.get_logger().debug("Hearing time: "+str(timer() - start)+" s")
        
        self.spectogram = cv2.resize(spectogram,(64,64),interpolation=cv2.INTER_LINEAR) / 255.0
        
        # Indicate that it is audio data
        self.spectogram[0] = -10.0
        
        mean = np.mean(np.abs(combine))
        
        if mean >= 0.7:
            self.audio_fear = 1.0
            
    def ext_emotion_callback(self,msg:Emotions):
        self.ext_emotion = msg
    
    def depth_image_callback(self,msg:Image):
        
        self.get_logger().info('I got depth image with format: %s' % msg.encoding)
        
        self.depth = self.bridge.imgmsg_to_cv2(msg)
        
        self.wait_for_depth = False
    
    def image_callback(self,msg:Image):
        
        self.get_logger().info('I got image with format: %s' % msg.encoding)
        
        self.image = self.bridge.imgmsg_to_cv2(msg)
        
        self.wait_for_img = False
    
    @codon.jit
    @staticmethod
    def points_callback_codon_code(points:np.ndarray):
        sorted_points = np.sort(points)
        
        min_points = np.mean(sorted_points[0:10])
        
        obstacle_detected = float(np.exp(-min_points*25))
        
        obstacle_detected = min(obstacle_detected,1.0)
        
        if obstacle_detected < 0.01:
            obstacle_detected = 0.0
            
        return obstacle_detected
      
    def points_callback(self, msg: PointCloud2):
        
        start = timer()
        
        # Read points from PointCloud2
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
            reshape_organized_cloud=True
        )
        
        self.last_points = points
        
        self.points_shape = (msg.width,msg.height,)
        
        self.obstacle_detected = MapMind.points_callback_codon_code(points)
         
        if self.obstacle_detected:    
            self.get_logger().info(f'Obstacle detected! {self.obstacle_detected}, time: {timer() - start} s')
            
    def sense_callabck(self,sense:PiezoSense):
        
        # should be rewritten 
        
        pin_states = sense.pin_state
        
        patting_sense = pin_states[4]
                
        if patting_sense:
            
            self.get_logger().debug('Patting detected')
                        
            self.pat_detected = 1.0
                                        
    @codon.jit            
    def emotion_state_calculate(self,emotions:list[float]):
                
        return  emotions[2]*320.0 + emotions[1]*-120.0 + emotions[3]*-40.0 + emotions[0]*-60.0 + emotions[4]*-20.0
    
    def send_ears_state(self,emotions:list[float]):
        
        max_id = 4
        
        if np.sum(emotions) >= 0.01:
            max_id = np.argmax(emotions[:4])
            
        self.get_logger().debug("Current emotion: "+str(ID_TO_EMOTION_NAME[max_id]))
            
        self.get_logger().debug("Sending angle to ears: "+str(self.emotions_angle[max_id]))
        
        angle:float = (self.emotions_angle[max_id]/180.0)*np.pi
        
        array=Float64MultiArray()
        
        array.data=[np.pi - angle, angle]
        
        self.ears_publisher.publish(array)
                
    
    def get_points_from_embedding(self,embedding:np.ndarray):
        results = self.obj_db.query(
            query_embeddings=embedding
        )
        
        output = []
        ids = []
        scores = []
        
        for id,distance,metadata in zip(results["ids"],results["distances"],results["metadatas"]):
            if distance < 0.1:     
                
                x = metadata['x']
                y = metadata['y']
                v = metadata['v']
                
                scores.append(v)
                           
                output.append(
                    (x,y)
                )
                ids.append(id)
        
        return np.array(output,dtype=np.float32),ids,np.array(scores,dtype=np.float32)
    
    def add_points_to_embedding(self,embedding:np.ndarray,points:np.ndarray,ids:list[str],scores:np.ndarray):
        # update or upsert points
        metadatas = []
        embeddings = []
        
        for point,score in zip(points,scores):
            ids.append(points)
            metadatas.append({
                "x":point[1],
                "y":point[2],
                "v":score
            })
            embeddings.append(embedding)
            
        self.obj_db.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def check_for_point(self,query:np.ndarray,points:np.ndarray):
        tree = KDTree(points,metric='euclidean')
        
        dist, ind = tree.query(query, k=3)
        
        if dist[0] < 0.1:
            return ind[0]
        
        return -1
    
    def add_point_to_map(self,point:np.ndarray,value:float):
        res = self.point_map_db.query(query_embeddings=[point],n_results=1)
        
        if res["distances"][0] < 0.1:
            id = res['ids'][0]
        else:
            id = str(uuid4())
        
        self.point_map_db.upsert(
                ids=[id],
                embeddings=[point],
                metadatas={
                    "val":value
                }
            )
        
    
    def point_seeking(self,score:float):
        # Update embeddings in the database with new score
        # TODO
        
        img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
        img_width = float(img.shape[0])
        img_height = float(img.shape[1])
                
        # Retrive points
        # Retrive points from visible objects
        res = self.detection_pipeline.run(img)
        
        img = cv2.resize(img,(64,64),interpolation=cv2.INTER_LINEAR) / 255.0
        
        # This way we indicate that it is image data
        img[0] = 10.0
        
        points_to_add_to_map = []
                
        if self.last_points is not None:
        
            for obj in res:
                # retrive points
                points,ids,scores = self.get_points_from_embedding(obj[0])
                
                scores = scores*0.6 + score*0.4
                                    
                # Add them to map, if positive
                x = obj[0]+obj[2]/2
                y = obj[1]+obj[3]/2
                
                x = int((x / img_width)*self.points_shape[0])
                y = int((y / img_height)*self.points_shape[1])
                
                a_point = self.last_points[ y*self.points_shape[0] + x ]
                
                distance = np.linalg.norm(a_point)
                
                c_point = (
                    self.position[0] + np.cos(self.yaw)*distance,
                    self.position[1] + np.sin(self.yaw)*distance     
                )
                
                c_point = np.array(c_point,dtype=np.float32)
                
                ind = self.check_for_point(c_point,points)
                # Add a new estimate point when it is not present
                if ind == -1:
                    np.append(scores,score)
                    np.append(points,n_point)
                    ids.append(str(uuid4()))
                else:
                    scores[ind] = score
                    
                self.add_points_to_embedding(obj[0],points,ids,scores)   
                
                for p,s in zip(points,scores):
                    points_to_add_to_map.append((p[0],p[1],s))     
        
        # get points associated with audio
        points,ids,scores = self.get_points_from_embedding(self.spectogram)
        
        scores = scores*0.6 + score*0.4
        
        # Update points        
        n_point = (self.position[0],self.position[1])
        
        n_id = self.check_for_point(n_point,points,ids,scores)
        
        if n_id == -1:
            np.append(scores,score)
            np.append(points,n_point)
            ids.append(str(uuid4()))
        else:
            scores[ind] = score

        self.add_points_to_embedding(self.spectogram,points,ids,scores)
        
        for p,s in zip(points,scores):
            points_to_add_to_map.append((p[0],p[1],s)) 
        
        # Move towards point with higher positive value
        # Get it from the map
        max_point = np.argmax(points_to_add_to_map)
        
        _max_point = PointValue2D()
        
        _max_point.x = max_point[0]
        _max_point.y = max_point[1]
        _max_point.v = max_point[2]
        
        self.value_point_pub.publish(_max_point)
        
        # Here is a algorithm for moving                        
        self.last_score = score
        
                
    def tick_func(self):
        # emotion validation pipeline
        
        # face detection
                
        # evaluate emotion states
        
        emotions = Emotions()
        
        self.emotion.happiness = self.pat_detected*10.0
        self.emotion.fear = self.obstacle_detected*1.0
        
        self.pat_detected = 0.0
        
        emotions_arr = [
            self.emotion.angry + self.ext_emotion.angry,
            self.emotion.fear + self.ext_emotion.fear,
            self.emotion.happiness + self.ext_emotion.happiness,
            self.emotion.uncertainty + self.ext_emotion.uncertainty,
            self.emotion.boredom + self.ext_emotion.boredom
        ]
        
        emotions.angry = emotions_arr[0]
        emotions.fear = emotions_arr[1]
        emotions.happiness = emotions_arr[2]
        emotions.uncertainty = emotions_arr[3]
        emotions.boredom = emotions_arr[4]
        
        self.emotion_pub.publish(emotions)
        
        score = self.emotion_state_calculate(emotions_arr)
                
        # send ears position
        self.send_ears_state(emotions_arr)
        
        self.point_seeking(score)
        
        # Think about inner workings of boredom
        # Move to the direction of point with higher score
        
        
    def tick_callback(self):
        
        if self.wait_for_img or self.wait_for_depth or self.spectogram is None:
            return
        
        self.timer.cancel()
        
        self.get_logger().info('Mind tick')
        
        start = timer()
        
        self.tick_func()
        
        end = timer()
        
        self.get_logger().info(f'Tick inference time {end - start} s')
        
        self.timer.reset()
        

def main(args=None):
    rclpy.init(args=args)

    map_mind = MapMind()

    rclpy.spin(map_mind)

    map_mind.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()