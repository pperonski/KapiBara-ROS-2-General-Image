#pragma once
#include <memory>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "kapibara_interfaces/msg/point_value2_d.hpp"

/*
A node that takes point with position and score, 
stores them and then move to point with highest 
score value.

I need to think about obstacle avoidance,
I wanted to create dead simple algorithm
for navigation but I am not really sure 
whether it is right way. 
So I guess some mapping is necessary nevertheless.
Particle filter sounds like intresting idea.

1. Generate points with random positions around 
robot.
2. Then we can use some Dijkstra algorithm or somehing.
I need to think it through.

Or I got diffrent idea, we are playing around with asotive,
memory so I guess potential field method would be much better
suited.

I guess I can write all of it in Python ...
*/

struct Point3D
{
  float x;
  float y;
  float z;
};

struct ValuePoint
{
  float x;
  float y;
  float v;
};

class ActractorNode : public rclcpp::Node
{
  public:
    ActractorNode();

  private:
    // robot position in space
    float x;
    float y;
    float yaw;
    bool waitForOdometry;

    float point_move_threshold;
    float point_min_scan;
    float wall_gain_avoid;

    // point cloud data
    // a size of the point 
    // cloud frame
    uint32_t width;
    uint32_t height;
    
    std::vector<ValuePoint> obstacles_points;

    std::vector<float> scan;

    std::vector<double> error_thresholds;
    std::vector<double> gain_thresholds;

    // value points
    std::vector<ValuePoint> value_points;

    void odometry_callaback(nav_msgs::msg::Odometry::SharedPtr msg);
    void cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void value_point_callback(kapibara_interfaces::msg::PointValue2D::SharedPtr msg);
    void timer_callback();

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;
    rclcpp::Subscription<kapibara_interfaces::msg::PointValue2D>::SharedPtr value_point_sub;

    rclcpp::TimerBase::SharedPtr timer_;
};
