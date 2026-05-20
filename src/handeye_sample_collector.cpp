/**
 * @file handeye_sample_collector.cpp
 * @brief Generic hand-eye sample collector using standard ROS Image + Pose topics.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

class HandeyeSampleCollector : public rclcpp::Node
{
public:
    HandeyeSampleCollector() : Node("handeye_sample_collector")
    {
        image_topic_ = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
        tool_pose_topic_ = declare_parameter<std::string>("tool_pose_topic", "/handeye/tool_pose");
        tool_pose_frame_id_ = declare_parameter<std::string>("tool_pose_frame_id", "base");
        output_dir_ = declare_parameter<std::string>("output_dir", "./handeye_data");
        image_prefix_ = declare_parameter<std::string>("image_prefix", "rgb");
        pose_csv_name_ = declare_parameter<std::string>("pose_csv_name", "poses.csv");
        save_service_name_ = declare_parameter<std::string>("save_service_name", "/handeye/save_sample");

        fs::create_directories(output_dir_);
        ensureCsvHeader();

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
            std::bind(&HandeyeSampleCollector::onImage, this, std::placeholders::_1));
        pose_sub_ = create_subscription<geometry_msgs::msg::Pose>(
            tool_pose_topic_, rclcpp::QoS(rclcpp::KeepLast(10)),
            std::bind(&HandeyeSampleCollector::onPose, this, std::placeholders::_1));
        save_srv_ = create_service<std_srvs::srv::Trigger>(
            save_service_name_,
            std::bind(&HandeyeSampleCollector::onSave, this, std::placeholders::_1,
                      std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "Generic hand-eye sample collector started");
        RCLCPP_INFO(get_logger(), " image_topic     : %s", image_topic_.c_str());
        RCLCPP_INFO(get_logger(), " tool_pose_topic : %s", tool_pose_topic_.c_str());
        RCLCPP_INFO(get_logger(), " pose_frame_id   : %s", tool_pose_frame_id_.c_str());
        RCLCPP_INFO(get_logger(), " output_dir      : %s", output_dir_.c_str());
        RCLCPP_INFO(get_logger(), " save_service    : %s", save_service_name_.c_str());
    }

private:
    static bool imageToBgr(const sensor_msgs::msg::Image& msg, cv::Mat& out, std::string& err)
    {
        if (msg.data.empty())
        {
            err = "empty image";
            return false;
        }

        const int width = static_cast<int>(msg.width);
        const int height = static_cast<int>(msg.height);
        if (width <= 0 || height <= 0)
        {
            err = "invalid image size";
            return false;
        }

        if (msg.encoding == sensor_msgs::image_encodings::BGR8)
        {
            cv::Mat view(height, width, CV_8UC3, const_cast<unsigned char*>(msg.data.data()),
                         static_cast<size_t>(msg.step));
            out = view.clone();
            return true;
        }
        if (msg.encoding == sensor_msgs::image_encodings::RGB8)
        {
            cv::Mat rgb(height, width, CV_8UC3, const_cast<unsigned char*>(msg.data.data()),
                        static_cast<size_t>(msg.step));
            cv::cvtColor(rgb, out, cv::COLOR_RGB2BGR);
            return true;
        }
        if (msg.encoding == sensor_msgs::image_encodings::MONO8)
        {
            cv::Mat mono(height, width, CV_8UC1, const_cast<unsigned char*>(msg.data.data()),
                         static_cast<size_t>(msg.step));
            cv::cvtColor(mono, out, cv::COLOR_GRAY2BGR);
            return true;
        }
        if (msg.encoding == sensor_msgs::image_encodings::BGRA8)
        {
            cv::Mat bgra(height, width, CV_8UC4, const_cast<unsigned char*>(msg.data.data()),
                         static_cast<size_t>(msg.step));
            cv::cvtColor(bgra, out, cv::COLOR_BGRA2BGR);
            return true;
        }
        if (msg.encoding == sensor_msgs::image_encodings::RGBA8)
        {
            cv::Mat rgba(height, width, CV_8UC4, const_cast<unsigned char*>(msg.data.data()),
                         static_cast<size_t>(msg.step));
            cv::cvtColor(rgba, out, cv::COLOR_RGBA2BGR);
            return true;
        }

        err = "unsupported image encoding: " + msg.encoding;
        return false;
    }

    static void quatToRpy(const geometry_msgs::msg::Quaternion& q_msg, double& roll, double& pitch,
                          double& yaw)
    {
        const double x = q_msg.x;
        const double y = q_msg.y;
        const double z = q_msg.z;
        const double w = q_msg.w;

        const double sinr_cosp = 2.0 * (w * x + y * z);
        const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        roll = std::atan2(sinr_cosp, cosr_cosp);

        const double sinp = 2.0 * (w * y - z * x);
        if (std::abs(sinp) >= 1.0)
        {
            pitch = std::copysign(M_PI / 2.0, sinp);
        }
        else
        {
            pitch = std::asin(sinp);
        }

        const double siny_cosp = 2.0 * (w * z + x * y);
        const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        yaw = std::atan2(siny_cosp, cosy_cosp);
    }

    void onImage(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (image_count_ == 0)
        {
            RCLCPP_INFO(get_logger(), "First image received: %ux%u, encoding=%s", msg->width,
                        msg->height, msg->encoding.c_str());
        }
        last_image_ = msg;
        ++image_count_;
    }

    void onPose(const geometry_msgs::msg::Pose::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pose_count_ == 0)
        {
            RCLCPP_INFO(get_logger(), "First tool pose received");
        }
        last_pose_ = msg;
        ++pose_count_;
    }

    void onSave(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        sensor_msgs::msg::Image::SharedPtr image;
        geometry_msgs::msg::Pose::SharedPtr pose;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            image = last_image_;
            pose = last_pose_;
        }

        if (!image)
        {
            res->success = false;
            res->message = "no image received";
            return;
        }
        if (!pose)
        {
            res->success = false;
            res->message = "no tool pose received";
            return;
        }

        cv::Mat bgr;
        std::string err;
        if (!imageToBgr(*image, bgr, err))
        {
            res->success = false;
            res->message = "image conversion failed: " + err;
            return;
        }

        const std::string image_name = buildImageName();
        const fs::path image_path = fs::path(output_dir_) / image_name;
        if (!cv::imwrite(image_path.string(), bgr))
        {
            res->success = false;
            res->message = "failed to write image: " + image_path.string();
            return;
        }

        if (!appendPoseRow(image_name, *pose))
        {
            res->success = false;
            res->message = "image saved, but failed to write pose csv";
            return;
        }

        res->success = true;
        res->message = "saved: " + image_path.string();
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
    }

    std::string buildImageName()
    {
        const auto ns = now().nanoseconds();
        std::ostringstream oss;
        oss << image_prefix_ << "_" << std::setw(6) << std::setfill('0') << sample_index_ << "_"
            << ns << ".png";
        ++sample_index_;
        return oss.str();
    }

    fs::path poseCsvPath() const
    {
        return fs::path(output_dir_) / pose_csv_name_;
    }

    void ensureCsvHeader()
    {
        const fs::path csv_path = poseCsvPath();
        if (fs::exists(csv_path))
        {
            return;
        }

        std::ofstream out(csv_path, std::ios::out);
        out << "image,wx,wy,wz,wrx,wry,wrz,qx,qy,qz,qw,stamp_ns,frame_id\n";
    }

    bool appendPoseRow(const std::string& image_name, const geometry_msgs::msg::Pose& pose)
    {
        std::ofstream out(poseCsvPath(), std::ios::app);
        if (!out.is_open())
        {
            return false;
        }

        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
        quatToRpy(pose.orientation, roll, pitch, yaw);

        out << image_name << std::setprecision(12) << "," << pose.position.x << ","
            << pose.position.y << "," << pose.position.z << "," << roll << "," << pitch << ","
            << yaw << "," << pose.orientation.x << "," << pose.orientation.y << ","
            << pose.orientation.z << "," << pose.orientation.w << "," << now().nanoseconds() << ","
            << tool_pose_frame_id_ << "\n";
        return true;
    }

    std::string image_topic_;
    std::string tool_pose_topic_;
    std::string tool_pose_frame_id_;
    std::string output_dir_;
    std::string image_prefix_;
    std::string pose_csv_name_;
    std::string save_service_name_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;

    std::mutex mutex_;
    sensor_msgs::msg::Image::SharedPtr last_image_;
    geometry_msgs::msg::Pose::SharedPtr last_pose_;
    std::size_t sample_index_{0};
    std::size_t image_count_{0};
    std::size_t pose_count_{0};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HandeyeSampleCollector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
