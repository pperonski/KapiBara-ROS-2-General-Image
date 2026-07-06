#include <memory>
#include <cmath>
#include <chrono>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "kapibara/actractor.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;


static double getYaw(double x,double y,double z,double w){
    // Yaw (z-axis rotation)
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    
    return std::atan2(siny_cosp, cosy_cosp);
}


ActractorNode::ActractorNode()
: Node("actractor")
{
   this->waitForOdometry = true;

    this->declare_parameter("point_move_threshold",0.25);
    this->declare_parameter("point_min_scan",0.25);
    this->declare_parameter("wall_gain_avoid",40.f);

    this->declare_parameter("yaw_thresholds",
        std::vector<double>({0.0})
    );
    this->declare_parameter("gain_thresholds",
        std::vector<double>({0.0})
    );

    this->error_thresholds = this->get_parameter("yaw_thresholds").as_double_array();
    this->gain_thresholds = this->get_parameter("gain_thresholds").as_double_array();
    this->wall_gain_avoid = this->get_parameter("wall_gain_avoid").as_double();

    this->point_move_threshold = this->get_parameter("point_move_threshold").as_double();
    this->point_min_scan = this->get_parameter("point_min_scan").as_double();

   this->odom_sub = this->create_subscription<nav_msgs::msg::Odometry>
        ("odom",10,std::bind(&ActractorNode::odometry_callaback,this,_1));
   this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>
        ("points",10,std::bind(&ActractorNode::cloud_callback,this,_1));
    this->value_point_sub = this->create_subscription<kapibara_interfaces::msg::PointValue2D>
        ("val_points",10,std::bind(&ActractorNode::value_point_callback,this,_1));
    this->twist_pub = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel",10);
    
   this->timer_ = this->create_wall_timer(100ms,
            std::bind(&ActractorNode::timer_callback,this));
}

void ActractorNode::odometry_callaback(nav_msgs::msg::Odometry::SharedPtr msg)
{
    this->waitForOdometry = false;

    this->x = msg->pose.pose.position.x;
    this->y = msg->pose.pose.position.y;
    
    this->yaw = getYaw(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
}

void ActractorNode::cloud_callback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    std::vector<Point3D> cloud_points;
    
    this->width = msg->width;
    this->height = msg->height;

    cloud_points.clear();
    cloud_points.reserve(this->width*this->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            
        if (!std::isnan(*iter_x) && !std::isnan(*iter_y) && !std::isnan(*iter_z)) {
            cloud_points.push_back({*iter_x, *iter_y, *iter_z});
        }else
        {
            cloud_points.push_back({100.0,100.0,100.0});
        }
    }

    // get a 2d scan from it

    std::vector<Point3D> lines;
    this->scan.clear();
    this->scan.reserve(this->width);

    for(size_t x=0;x<this->width;++x)
    {
        float scan_value = 0;

        for(size_t y=0;y<this->height;++y)
        {
            Point3D point = cloud_points[y*this->width + x];

            if(std::isinf(point.x) && point.x < 0)
            {
                point.x = 0.f;
            }

            if(std::isinf(point.y) && point.y < 0)
            {
                point.y = 0.f;
            }

            if(std::isinf(point.z) && point.z < 0)
            {
                point.z = 0.f;
            }

            float dist = std::sqrt(
                point.x*point.x + 
                point.y*point.y + 
                point.z*point.z
            );

            scan_value += dist;
        }
        
        scan_value /= this->height;

        this->scan.push_back(scan_value);
    }

}

void ActractorNode::value_point_callback(kapibara_interfaces::msg::PointValue2D::SharedPtr msg)
{

    if( this->value_points.size() == 64 )
    {
        RCLCPP_INFO(this->get_logger(),"Value points buffer full.");
        return;
    }

    // check whether point isn't aleardy presents

    ValuePoint point;

    point.x = msg->x;
    point.y = msg->y;
    point.v = msg->value;

    for( const ValuePoint& p : this->value_points )
    {
        float distance = std::sqrt(
            (p.x - point.x)*(p.x - point.x) +
            (p.y - point.y)*(p.y - point.y)
        );

        // point is in set
        if( distance < 0.25 )
        {
            RCLCPP_INFO(this->get_logger(),"Point is aleardy in set!");
            return;
        }
    }

    // if not add it 
    RCLCPP_INFO(this->get_logger(),"Adding new point: %f %f %f", point.x,point.y,point.v);
    this->value_points.push_back(point);
}

void ActractorNode::timer_callback()
{
    // wait for odometry message
    if( this->waitForOdometry )
    {
        RCLCPP_INFO(this->get_logger(),"Waitting for odometry.");
        return;
    }

    if( this->value_points.empty() )
    {
        RCLCPP_INFO(this->get_logger(),"Waitting for value points.");
        return;
    }

    geometry_msgs::msg::Twist twist;

    ValuePoint max_value_point;

    // get value point with biggest value and move towards it

    std::sort(
                this->value_points.begin(),
                this->value_points.end(),
                [](
                    const ValuePoint& a,
                    const ValuePoint& b
                ){
                    return a.v > b.v;
                }
            );
    
    max_value_point = this->value_points[0];

    RCLCPP_INFO(this->get_logger(),"Max value point: %f %f %f", 
            max_value_point.x, max_value_point.y, max_value_point.v);

    // move towards it

    if( 
        abs(max_value_point.x - this->x)+
        abs(max_value_point.y - this->y)
        > 0.25
    )
    {

        float target_yaw = std::atan2(
            max_value_point.y - this->y,
            max_value_point.x - this->x
        );

        float error = target_yaw - this->yaw;

        error = std::atan2(
            std::sin(error),
            std::cos(error)
        );

        RCLCPP_INFO(this->get_logger(),"Yaw error: %f ",error);

        float a_error = abs(error);

        float gain = 0;
        float linear = 0;
        
        size_t i=0;

        for(;i<this->error_thresholds.size();i++)
        {
            if( a_error > this->error_thresholds[i] )
            {
                gain = this->gain_thresholds[i];
                break;
            }
        }

        linear = 0.1;

        twist.linear.x = linear;
        twist.angular.z = gain*error;

        // wall detection error
        // we take into account line scans

        if( !this->scan.empty() )
        {
            float error = 0;
            float min_value = 9999999;
            float max_value = -9999999;

            for(auto it = this->scan.end()-1;
                it != this->scan.begin();
                it--)
            {
                if( *it <= this->point_min_scan )
                {
                    error += 1.0;
                }

                if( *it < min_value )
                {
                    min_value = *it;
                }

                if( *it > max_value )
                {
                    max_value = *it;
                }
            }

            error /= this->width;

            if( error > this->point_move_threshold )
            {
                twist.linear.x = 0.0f;
            }

            RCLCPP_INFO(this->get_logger(),"Wall error: %f",error);
            RCLCPP_INFO(this->get_logger(),"Min scan: %f",min_value);
            RCLCPP_INFO(this->get_logger(),"Max scan: %f",max_value);

            if( error > 0.1 )
            {
                twist.angular.z = error*this->wall_gain_avoid;
            }
        }

    }
    else
    {
        twist.angular.z = 0.0;
        twist.linear.x = 0.0;
        RCLCPP_INFO(this->get_logger(),"Robot is at target point!");
    }

    // update points
    RCLCPP_INFO(this->get_logger(),"Update points!");
    for(    
            auto it = this->value_points.begin();
            it  != this->value_points.end();
        )
    {
        it->v *= 0.99;

        if( abs(it->v) < 0.2 )
        {
            this->value_points.erase(it);
        }
        else
        {
            it++;
        }
    }

    RCLCPP_INFO(this->get_logger(),"Publish twist, l: %f, v: %f",
            twist.linear.x, twist.angular.z );
    this->twist_pub->publish(twist);

}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ActractorNode>());
  rclcpp::shutdown();
  return 0;
}