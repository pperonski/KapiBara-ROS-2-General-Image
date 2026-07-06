#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.neighbors import KDTree
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint

from kapibara_interfaces.msg import PointValue2D
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import Quaternion
from cv_bridge import CvBridge

import open3d as o3d
import open3d.core as o3c
import numpy as np
from timeit import default_timer as timer

def get_yaw_from_quaternion(q: Quaternion) -> float:
    """
    Converts a ROS 2 geometry_msgs Quaternion to Euler yaw (in radians).
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return yaw

class Atractor(Node):

    def __init__(self):
        super().__init__('atractor')
        
        self.declare_parameter("tick_time",0.02)
        
        self.timer_period = self.get_parameter('tick_time').get_parameter_value().double_value  # seconds
        self.timer = self.create_timer(self.timer_period, self.tick_callback)
        
        self.twist_pub = self.create_publisher(
            Twist,
            "cmd_vel",
            10
        )
        
        self.value_point_sub = self.create_subscription(
            PointValue2D,
            'val_points',
            self.value_point_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
                
        self.yaw = 0.0
        self.position = None
        
        self.target_yaw = 0.0
                
        self.c_device = o3c.Device("CPU:0")
        self.point_map = o3d.t.geometry.PointCloud(self.c_device)
        
        self.target_points = o3d.t.geometry.PointCloud(self.c_device)
        
        # cost map of size 9 * 9
        self.costmap_grid_size = 0.25
        self.costmap = np.zeros((9,9),dtype=np.float32)
            
    def value_point_callback(self,msg:PointValue2D):
        
        point = np.array([
            msg.x,
            msg.y,
            0
        ])
        value = msg.value
        
        n_cloud = o3d.t.geometry.PointCloud(self.c_device)
        
        n_cloud.point.positions = o3c.Tensor(point.reshape(1,3), o3c.float32)
        n_cloud.point.weights = o3c.Tensor(
            np.array([value],dtype=np.float32).reshape(1,1),
            o3c.float32
        )
        
        if self.target_points.is_empty():
            self.target_points = n_cloud.clone()
        else:
            self.target_points.append(n_cloud)
            self.target_points.voxel_down_sample( voxel_size = 0.25 )
    
    def odom_callback(self,msg:Odometry):
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            0.0
            ])
        
        self.yaw = get_yaw_from_quaternion(msg.pose.pose.orientation)
        
    def calculate_point_costmap_strength(self,x,y,cloud_points):
        # x, y costmap positions
        
        width = self.costmap.shape[0]
        height = self.costmap.shape[1]
        
        real_x = (x - int(width/2))*self.costmap_grid_size
        real_y = (y - int(height/2))*self.costmap_grid_size
        
        real_pos = np.array([real_x,real_y],dtype=np.float32)
        
        cloud_positions = cloud_points.points.position.numpy()
        cloud_weights = cloud_points.points.weights.numpy()
        
        dist = real_pos - cloud_positions
        
        force = (1.0/np.linalg.norm(dist,axis=0))*cloud_weights
        
        self.costmap[x][y] = np.mean(force)

    
    def tick_callback(self):
        if self.position is None:
            return
        if self.target_points.is_empty():
            return
                
        max_point_id = np.argmax(self.target_points.point.weights)
        
        max_point = self.target_points.point.positions[max_point_id].numpy()
        max_point_value = self.target_points.point.weights[max_point_id].numpy()
        
        position = self.position
        
        bbox = o3d.t.geometry.AxisAlignedBoundingBox()
        
        bbox.min_bound = o3c.Tensor(np.ones(3)*-5.0 + position, o3c.float32)
        bbox.max_bound = o3c.Tensor(np.ones(3)*5.0 + position, o3c.float32)
        
        points = self.target_points.crop(bbox)
        
        points.point.positions = points.point.positions.extend(max_point)
        points.point.weights = points.point.weights.extend(max_point_value)
        
        twist = Twist()
        
        for y in range(9):
            for x in range(9):
                self.calculate_point_costmap_strength(
                    x,
                    y,
                    points
                )
                
        position_error = np.abs(max_point[0]-self.position[0])+np.abs(max_point[1]-self.position[1])        
                
        if self.costmap.max() != 0:
        
            # find a value with biggest value in costmap
            
            biggest_force_index = np.unravel_index(self.costmap.argmax(),self.costmap.shape)
            
            coord_x = biggest_force_index[0] - int(self.costmap.shape[0]/2)
            coord_y = biggest_force_index[1] - int(self.costmap.shape[1]/2)
            
            target_yaw = np.arctan2(
                coord_x,
                coord_y
            )
            
            e = target_yaw - self.yaw
            
            yaw_error = np.arctan2(
                np.sin(e),
                np.cos(e)
            )
            
            self.get_logger().info(f"Yaw error: {yaw_error}, current yaw: {self.yaw}")
            
            if np.abs(yaw_error) > 0.2:
                twist.angular = 1.5
            elif position_error > 0.25:
                twist.linear = 0.5
            
        self.twist_pub.publish(twist)
            
        self.target_points.point.weights *= 0.999
        to_remove = np.where( (self.target_points.point.weights < 0.01) 
                             & (self.target_points.point.weights > -0.01) )[0]
        
        self.target_points = self.target_points.select_by_index(
            indices = to_remove,
            invert = True
        )
        
        
def main(args=None):
    rclpy.init(args=args)
    node = Atractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
