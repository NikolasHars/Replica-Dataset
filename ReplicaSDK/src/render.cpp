// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include <array>
#include <map>
#include <vector>
#include <stdio.h>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>

#include "GLCheck.h"
#include "MirrorRenderer.h"

#include <sstream>


using namespace std;
using namespace Eigen;

#define DEG2RAD 3.14159265 / 180.0

std::string YAML_FILE = "/cluster/project/infk/courses/252-0579-00L/group04/processed_data/conf.yaml";
auto CONFIG = YAML::LoadFile(YAML_FILE);

Eigen::Matrix4d getMatrixFromCSVLine(std::vector< std::string >& line_of_file) {
  Eigen::Matrix4d hom_pose;
  hom_pose << stod(line_of_file[0]), stod(line_of_file[1]), stod(line_of_file[2]), stod(line_of_file[3]),
              stod(line_of_file[4]), stod(line_of_file[5]), stod(line_of_file[6]), stod(line_of_file[7]),
              stod(line_of_file[8]), stod(line_of_file[9]), stod(line_of_file[10]), stod(line_of_file[11]),
              stod(line_of_file[12]), stod(line_of_file[13]), stod(line_of_file[14]), stod(line_of_file[15]);
  return hom_pose;
} 

std::vector< Eigen::Matrix4d > readCSV(std::string filename) {
  // read the csv file
  std::vector< Eigen::Matrix4d> content;
	std::vector<std::string> row;
	std::string line, word; 
  int i = 0;
 
	std::fstream file (filename, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			std::stringstream str(line);
      i = 0;
			while(getline(str, word, ',')) {
				if (i > 0)
          row.push_back(word);
        ++i; // ignore first word which is a string
      }

      Eigen::Matrix4d tmp_matrix = getMatrixFromCSVLine(row);
			content.push_back(tmp_matrix);
		}
    file.close();
    return content;
	}
	else
		cout<<"Could not open the file\n";
    return content;
}


/**
  theta_x = atan2(r32, r33)
  theta_y = atan2(-r31, sqrt(r32^2 + r33^2))
  theta_z = atan2(r21, r11)
*/
std::array<double, 3> getAngleArray(Eigen::Matrix4d& hom_pose) {
  double theta_x = atan2(hom_pose(2,1), hom_pose(2,2));
  double theta_y = atan2(-hom_pose(2,0), sqrt(hom_pose(0,0) * hom_pose(0,0) + hom_pose(1,0) * hom_pose(1,0)));
  double theta_z = atan2(hom_pose(1,0), hom_pose(0,0));
  return std::array<double, 3> {theta_x, theta_y, theta_z};
}

std::array<double, 3> getTranslationArray(Eigen::Matrix4d& hom_pose) {
  return std::array<double, 3> {hom_pose(0,3), hom_pose(1,3), hom_pose(2,3)};
}


void saveData(string fileName, MatrixXd  matrix, string img_name)
{
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", ",", "", "");
 
    ofstream file;
    file.open(fileName, std::ios_base::app);
    if (file.is_open())
    {
        file << img_name << "," << matrix.format(CSVFormat) << "\n";
        file.close();
    }
}

// Generates two Eigen vectors that are perpendicular to a specified vector and to each other
void generatePerpendicularVectors(const Eigen::Vector3d& vector, Eigen::Vector3d& v1, Eigen::Vector3d& v2)
{
    // Generate a random vector
    Eigen::Vector3d random_vector = Eigen::Vector3d::Random();

    // Project the random vector onto the plane perpendicular to the specified vector
    Eigen::Vector3d projected_vector = random_vector - random_vector.dot(vector) * vector;

    // Set the first perpendicular vector to the normalized projected vector
    v1 = projected_vector.normalized();

    // Set the second perpendicular vector to the cross product of the specified vector and the first perpendicular vector
    v2 = vector.cross(v1).normalized();
}

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Affine3d rx =
      Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry =
      Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz =
      Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

// Generates a random rotation matrix around two perpendicular axes
void randomRotation(
  const float rotation_angle, 
  const Eigen::Vector3d& axis1, 
  const Eigen::Vector3d& axis2,
  const Eigen::Vector3d& translation_vector,
  Eigen::Matrix4d& cam_matrix,
  Eigen::Matrix4d& transformation_matrix)
{
    // Generate a random angle between 0 and random_angle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-rotation_angle, rotation_angle);
    
    // generate random angles for the two axes
    const double angle_1 = dis(gen)/5;
    const double angle_2 = dis(gen);

    // Create a rotation matrix from the first axis and angle
    //Eigen::AngleAxisd rotation1(angle_1, axis1);
    Eigen::AngleAxisd rotation1(angle_1, translation_vector.normalized().cross(Eigen::Vector3d::UnitZ()));

    // Create a rotation matrix around the second axis
    // Eigen::AngleAxisd rotation2(angle_2, axis2);
    Eigen::AngleAxisd rotation2(angle_2, Eigen::Vector3d::UnitZ());

    // Combine the rotation matrices
    Eigen::Matrix3d rotation_matrix = (rotation2 * rotation1).toRotationMatrix();

    // Invert the rotation matrix to be able to rotate the camera back to the original path -> no strong deviation from path
    // Eigen::Matrix3d inv_rotation_matrix3d = rotation_matrix.inverse();

    // Create a 3D affine transformation from the rotation matrix and translation vector
    Eigen::Transform<double, 3, Eigen::Affine> affine_transform;
    affine_transform = rotation_matrix;
    affine_transform.translation() = translation_vector;

    // Convert the affine transformation to a 4x4 homogeneous transformation matrix
    transformation_matrix = affine_transform.matrix();

    // // Rotate the camera back to the original path while maintaining the new incremental transformation
    // transformation_matrix *= inv_rotation_matrix4d;

    // // Create a 3D affine transformation from the inverse rotation matrix and 0 translation vector
    // Eigen::Transform<double, 3, Eigen::Affine> inv_affine_transform;
    // inv_affine_transform = inv_rotation_matrix3d;
    // inv_affine_transform.translation() = Eigen::Vector3d::Zero();

    // inv_rotation_matrix4d = inv_affine_transform.matrix();

    return;
}

void saveFloat(string fileName, float exposure)
{
    ofstream file;
    file.open(fileName, std::ios_base::app);
    if (file.is_open())
    {
        file << "exposure: " << exposure << "\n";
        file.close();
    }
}

void saveYAML(string fileName, double value, string value_desc)
{
    ofstream file;
    file.open(fileName, std::ios_base::app);
    if (file.is_open())
    {
        file << value_desc << ": " << value << "\n";
        file.close();
    }
}

/**
 * @brief argmax of 2 float numbers
 * @param first first float number 
 * @param second second float number
 * @return index of greater number
 */
int argmax(float first, float second) {
  if (first > second) return 0;
  return 1;
}


// Extracts the intrinsic camera parameters and stores them in a YAML file
void storeCamParams(int w, int h, double fu, double fv, double u0, double v0, double zNear, double zFar) {
    Eigen::Matrix4d M;
    M << fu, 0.0, u0, 0.0,
         0.0, fv, v0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0;
    saveData("cam_params.csv", M, "cam_params");
    saveYAML("cam_params.yaml", w, "width");
    saveYAML("cam_params.yaml", h, "height");
    saveYAML("cam_params.yaml", zNear, "zNear");
    saveYAML("cam_params.yaml", zFar, "zFar");
}


int main(int argc, char* argv[]) {
  // Load all scene-related things
  std::string data_path = CONFIG["data_path"].as<std::string>();
  std::string scene_name = std::string(argv[1]);
  const std::string meshFile = data_path + scene_name + "/mesh.ply";
  const std::string atlasFolder = data_path + scene_name + "/textures";
  const std::string surfaceFile = data_path + scene_name + "/glass.sur";

  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));
  ASSERT(pangolin::FileExists(surfaceFile));

  // width + height
  const int width = CONFIG["width"].as<int>();
  const int height = CONFIG["height"].as<int>();

  auto scene_paths = CONFIG["scene_to_rooms"].as< std::map< std::string, std::vector< std::vector<float> > > > ();
  std::vector< std::vector<float>> paths = scene_paths.at(scene_name);

  bool renderDepth = true;
  float depthScale = 65535.0f * 0.1f;

  // Number of images
  const int numFrames = CONFIG["num_imgs"].as<int>();
  float float_num = static_cast<float>(numFrames);

  // Exposure settings
  float exposure;
  bool set_exposure = false;
  auto exposure_settings = CONFIG["exposures"].as<std::map<std::string, float>>();
  if (exposure_settings.count(scene_name) == 1) {
    exposure = exposure_settings[scene_name];
    set_exposure = true;
  }

  // Check if there are poses to use in a pose.csv file
  bool use_poses = false;
  auto csv_file_map_map = CONFIG["csv_files"].as< std::map < std::string, std::map < std::string, std::string > > > ();
  std::vector<Eigen::Matrix4d> csv_file_poses;
  std::string csv_file_name_map = CONFIG["csv_file_name"].as< std::string > ();
  if (csv_file_map_map.count(scene_name) == 1) {
    std::map < std::string, std::string > csv_file_map = csv_file_map_map[scene_name];
    if (csv_file_map.count(csv_file_name_map) == 1) {
      std::string csv_file_name = csv_file_map[csv_file_name_map];
      csv_file_poses = readCSV(csv_file_name);
      use_poses = true;
    }
  }

  // Get rotation angle if any
  float rotation_angle = CONFIG["rotation_angle"].as<float>() * DEG2RAD;
  bool do_rotation = CONFIG["do_rotation"].as<bool>();


  // Output folder and pose file
  std::string name = scene_name;
  std::string pose_file_name = name + "_poses.csv";

  // Setup EGL
  EGLCtx egl;

  egl.PrintInformation();

  if(!checkGLVersion()) {
    return 1;
  }

  //Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  // Depth buffer
  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder);

  // Set exposure
  if (set_exposure) {
    ptexMesh.SetExposure(exposure);
  }
  saveFloat("exposure.txt", ptexMesh.Exposure());

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Setup a camera
  pangolin::OpenGlRenderState s_cam;
  Eigen::Matrix4d T_camera_world, T_affine, M_intr;
  Eigen::Matrix3d M_cam;
  Eigen::Vector3d incremental_translation, axis1, axis2;
  Eigen::Affine3d T_affine_3d;

  if (use_poses) {
    // get number of poses 
    int num_poses = csv_file_poses.size();
    int iterations = num_poses - 1;

    Eigen::Matrix4d pose_initial, pose_final;
    double x_increment, y_increment, z_increment;
    double x_angle_increment, y_angle_increment, z_angle_increment;
    std::array<double, 3> angles_initial, angles_final;
    std::array<double, 3> translations_initial, translations_final;

    // For each iteration we have to get the angle and translation between the two poses
    pose_initial = csv_file_poses.at(0);
    translations_initial = getTranslationArray(pose_initial);
    angles_initial = getAngleArray(pose_initial);
    for (int iter=0; iter < iterations; ++iter) {
      // Get final pose
      pose_final = csv_file_poses.at(iter+1);
      translations_final = getTranslationArray(pose_final);
      angles_final = getAngleArray(pose_final);

      saveData("hallo.txt", pose_initial, "hallo");

      // Compute increments
      x_increment = (translations_final[0] - translations_initial[0]) / float_num;
      y_increment = (translations_final[1] - translations_initial[1]) / float_num;
      z_increment = (translations_final[2] - translations_initial[2]) / float_num;
      x_angle_increment = (angles_final[0] - angles_initial[0]) / float_num;
      y_angle_increment = (angles_final[1] - angles_initial[1]) / float_num;
      z_angle_increment = (angles_final[2] - angles_initial[2]) / float_num;

      // Set View matrix
      s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrixRDF_BottomLeft(
            width,
            height,
            width / 2.0f,
            width / 2.0f,
            (width - 1.0f) / 2.0f,
            (height - 1.0f) / 2.0f,
            0.1f,
            100.0f),
        pangolin::ModelViewLookAt(
          0, 0, 0, 
          1, 0, 0, 
          pangolin::AxisNegZ));

      // Get Intrinsics for config: (width, height, fx, fy, cx, cy, near, far)
      storeCamParams(width, height, width / 2.0f, width / 2.0f, (width - 1.0f) / 2.0f, (height - 1.0f) / 2.0f, 0.1f, 50.0f);

      T_camera_world = pose_initial;
      s_cam.SetModelViewMatrix(T_camera_world);

      Eigen::Matrix4d projMatrix = s_cam.GetProjectionMatrix();
      saveData("projMatrix.txt", projMatrix, "worldmatrix");

      // render some frames
      for (auto i = 0; i < numFrames; i++) {
        std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
        std::cout.flush();

        // Render
        frameBuffer.Bind();
        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, width, height);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glEnable(GL_CULL_FACE);

        ptexMesh.Render(s_cam);

        glDisable(GL_CULL_FACE);

        glPopAttrib(); // GL_VIEWPORT_BIT
        frameBuffer.Unbind();

        for (size_t i = 0; i < mirrors.size(); i++) {
          MirrorSurface& mirror = mirrors[i];
          // capture reflections
          mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

          frameBuffer.Bind();
          glPushAttrib(GL_VIEWPORT_BIT);
          glViewport(0, 0, width, height);

          // render mirror
          mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

          glPopAttrib(); //GL_VIEWPORT_BIT
          frameBuffer.Unbind();
        }

        // Download and save
        render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

        char filename[1000];
        snprintf(filename, 1000, "imgs/%03d_frame_%06d.jpg", iter, i);

        saveData(pose_file_name, T_camera_world, std::string(filename));

        pangolin::SaveImage(
            image.UnsafeReinterpret<uint8_t>(),
            pangolin::PixelFormatFromString("RGB24"),
            std::string(filename));

        // Render depth
        if (renderDepth) {
          // render depth
          depthFrameBuffer.Bind();
          glPushAttrib(GL_VIEWPORT_BIT);
          glViewport(0, 0, width, height);
          glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

          glEnable(GL_CULL_FACE);

          ptexMesh.RenderDepth(s_cam, depthScale);

          glDisable(GL_CULL_FACE);

          glPopAttrib(); //GL_VIEWPORT_BIT
          depthFrameBuffer.Unbind();

          depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

          // convert to 16-bit int
          for(size_t i = 0; i < depthImage.Area(); i++)
              depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

          snprintf(filename, 1000, "labels/%03d_depth_%06d.png", iter, i);
          pangolin::SaveImage(
              depthImageInt.UnsafeReinterpret<uint8_t>(),
              pangolin::PixelFormatFromString("GRAY16LE"),
              std::string(filename), true, 34.0f);
        }

      // Apply incremental translation (+ rotation if enabled)
      T_affine_3d = create_rotation_matrix(
        angles_initial[0] + (i+1)*x_angle_increment, 
        angles_initial[1] + (i+1)*y_angle_increment, 
        angles_initial[2] + (i+1)*z_angle_increment
      );

      T_affine_3d.translation() = Eigen::Vector3d(
        translations_initial[0] + (i+1)*x_increment, 
        translations_initial[1] + (i+1)*y_increment, 
        translations_initial[2] + (i+1)*z_increment
      );

      T_camera_world = T_affine_3d.matrix(); 

      // Set View matrix
      s_cam.SetModelViewMatrix(T_camera_world);
      }
      std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;
      // Afterwards, we have to update the initial pose
      pose_initial = pose_final; 
      translations_initial = translations_final;
      angles_initial = angles_final;
    }
  }
  else {
    float x_increment, y_increment, z_increment;
    int argmax_idx;
    for (int idx=0; idx < paths.size(); ++idx) {
      argmax_idx = argmax(paths.at(idx)[2], paths.at(idx)[5]);
      // Get incremental translation from the poses: (x_max - x_min) / num_frames
      x_increment = (paths.at(idx)[3*argmax_idx] - paths.at(idx)[3-3*argmax_idx]) / float_num; 
      y_increment = (paths.at(idx)[1+3*argmax_idx] - paths.at(idx)[4-3*argmax_idx]) / float_num; 
      z_increment = (paths.at(idx)[2+3*argmax_idx] - paths.at(idx)[5-3*argmax_idx]) / float_num; 
      
      // Get incremental translation vector (for each step, apply this translation)
      incremental_translation = Eigen::Vector3d(-x_increment, -y_increment, -z_increment);

      // Get perpendicular axes to the incremental translation vector
      generatePerpendicularVectors(incremental_translation, axis1, axis2);

      // Incremental Transformation Matrix
      T_affine << 1, 0, 0, -x_increment,
                  0, 1, 0, -y_increment,
                  0, 0, 1, -z_increment,
                  0, 0, 0, 1;  

      // // Back rotation matrix (initialize as identity)
      // T_backrot << 1, 0, 0, 0,
      //              0, 1, 0, 0,
      //              0, 0, 1, 0,
      //              0, 0, 0, 1;

      // Set View matrix
      s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrixRDF_BottomLeft(
            width,
            height,
            width / 2.0f,
            width / 2.0f,
            (width - 1.0f) / 2.0f,
            (height - 1.0f) / 2.0f,
            0.1f,
            100.0f),
        pangolin::ModelViewLookAt(
          paths.at(idx)[3-3*argmax_idx], paths.at(idx)[4-3*argmax_idx], paths.at(idx)[5-3*argmax_idx], 
          paths.at(idx)[3*argmax_idx], paths.at(idx)[1+3*argmax_idx], paths.at(idx)[2+3*argmax_idx], 
          pangolin::AxisNegZ));

      // Get Intrinsics for config: (width, height, fx, fy, cx, cy, near, far)
      storeCamParams(width, height, width / 2.0f, width / 2.0f, (width - 1.0f) / 2.0f, (height - 1.0f) / 2.0f, 0.1f, 50.0f);

      // Start at some origin (set by input values)
      T_camera_world = s_cam.GetModelViewMatrix();

      // Extract projection matrix as pangolin::OpenGlMatrix
      Eigen::Matrix4d projMatrix = s_cam.GetProjectionMatrix();

      // // Transform projection matrix to Eigen::Matrix4d
      // Eigen::Matrix4d eigenProjMatrix;
      // for (int i = 0; i < 16; ++i) {
      //   eigenProjMatrix(i) = projMatrix.m[i];
      // }

      saveData("projMatrix.txt", projMatrix, "worldmatrix");

      // render some frames
      for (auto i = 0; i < numFrames; i++) {
        std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
        std::cout.flush();

        // Render
        frameBuffer.Bind();
        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, width, height);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glEnable(GL_CULL_FACE);

        ptexMesh.Render(s_cam);

        glDisable(GL_CULL_FACE);

        glPopAttrib(); // GL_VIEWPORT_BIT
        frameBuffer.Unbind();

        for (size_t i = 0; i < mirrors.size(); i++) {
          MirrorSurface& mirror = mirrors[i];
          // capture reflections
          mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

          frameBuffer.Bind();
          glPushAttrib(GL_VIEWPORT_BIT);
          glViewport(0, 0, width, height);

          // render mirror
          mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

          glPopAttrib(); //GL_VIEWPORT_BIT
          frameBuffer.Unbind();
        }

        // Download and save
        render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

        char filename[1000];
        snprintf(filename, 1000, "imgs/%03d_frame_%06d.jpg", idx, i);

        saveData(pose_file_name, T_camera_world, std::string(filename));

        pangolin::SaveImage(
            image.UnsafeReinterpret<uint8_t>(),
            pangolin::PixelFormatFromString("RGB24"),
            std::string(filename));

        // Render depth
        if (renderDepth) {
          // render depth
          depthFrameBuffer.Bind();
          glPushAttrib(GL_VIEWPORT_BIT);
          glViewport(0, 0, width, height);
          glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

          glEnable(GL_CULL_FACE);

          ptexMesh.RenderDepth(s_cam, depthScale);

          glDisable(GL_CULL_FACE);

          glPopAttrib(); //GL_VIEWPORT_BIT
          depthFrameBuffer.Unbind();

          depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

          // convert to 16-bit int
          for(size_t i = 0; i < depthImage.Area(); i++)
              depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

          snprintf(filename, 1000, "labels/%03d_depth_%06d.png", idx, i);
          pangolin::SaveImage(
              depthImageInt.UnsafeReinterpret<uint8_t>(),
              pangolin::PixelFormatFromString("GRAY16LE"),
              std::string(filename), true, 34.0f);
        }

        // First transform transformation matrix, then Move the camera to desired position
        if (do_rotation) {
          // Rotate the camera
          randomRotation(rotation_angle, axis1, axis2, incremental_translation, T_camera_world, T_affine);
        }

        // Apply incremental translation (+ rotation if enabled)
        T_camera_world = T_camera_world * T_affine;

        // Set View matrix
        s_cam.SetModelViewMatrix(T_camera_world);
      }
      std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;
    }
  }
  return 0;
}
