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
#include <map>
#include <vector>
#include <stdio.h>

#include "GLCheck.h"
#include "MirrorRenderer.h"

using namespace std;
using namespace Eigen;

#define PI 3.14159265

std::string YAML_FILE = "/cluster/project/infk/courses/252-0579-00L/group04/processed_data/conf.yaml";
auto CONFIG = YAML::LoadFile(YAML_FILE);

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


int argmax(float first, float second) {
  if (first > second) {
    return 0;
  } 
  else {
    return 1;
  }
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
  float exposure = CONFIG["exposure"].as<float>();
  bool set_exposure = CONFIG["set_exposure"].as<bool>();

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

  // Normal buffer
  pangolin::GlTexture normalTexture(width, height, GL_RGB32F, false, 0, GL_RGB, GL_FLOAT, 0);
  pangolin::GlRenderBuffer normalRenderBuffer(width, height);
  pangolin::GlFramebuffer normalFrameBuffer(normalTexture, normalRenderBuffer);
  normalFrameBuffer.AttachColour(normalTexture);

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


  // Shader for rendering the normal map
  // const std::string normalShader =
  //   "#version 330\n"
  //   "uniform mat4 u_MVP;"
  //   "uniform mat4 u_MV;"
  //   "in vec3 v_Position;"
  //   "in vec3 v_Normal;"
  //   "out vec3 out_Normal;"
  //   "void main()"
  //   "{"
  //   "  gl_Position = u_MVP * vec4(v_Position, 1.0);"
  //   "  out_Normal = normalize((u_MV * vec4(v_Normal, 0.0)).xyz);"
  //   "}";

  // pangolin::GlSlProgram normalShaderProgram = pangolin::GlSlProgram::Compile(
  //   normalShader, "#version 330\nout vec3 out_Normal;");


  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Setup a camera
  pangolin::OpenGlRenderState s_cam;
  Eigen::Matrix4d T_camera_world, T_new_old, T_new_old_inv;
  Eigen::Matrix3d M_cam;

  // Eigen::MatrixXf normalImage(width, height * 3);

  float x_increment, y_increment, z_increment;
  //float x_max, x_min, y_max, y_min, z_max, z_min; 
  int argmax_idx;
  for (int idx=0; idx < paths.size(); ++idx) {
    argmax_idx = argmax(paths.at(idx)[2], paths.at(idx)[5]);
    // Get incremental translation from the poses: (x_max - x_min) / num_frames
    x_increment = (paths.at(idx)[3*argmax_idx] - paths.at(idx)[3-3*argmax_idx]) / float_num; 
    y_increment = (paths.at(idx)[1+3*argmax_idx] - paths.at(idx)[4-3*argmax_idx]) / float_num; 
    z_increment = (paths.at(idx)[2+3*argmax_idx] - paths.at(idx)[5-3*argmax_idx]) / float_num; 
    
    // Only translate view
    T_new_old << 1, 0, 0, x_increment,
                 0, 1, 0, y_increment,
                 0, 0, 1, z_increment,
                 0, 0, 0, 1;
    
    T_new_old_inv = T_new_old.inverse();

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

    // Start at some origin (set by input values)
    T_camera_world = s_cam.GetModelViewMatrix();
    //std::cout << T_camera_world.col(3) << std::endl;
    //saveData("projection.csv", M_cam, std::string("INTRINSICS"));
    //T_camera_world.col(3) = Eigen::Vector4d(paths.at(idx)[0], paths.at(idx)[1], paths.at(idx)[2], 1);
    //#s_cam.SetModelViewMatrix(T_camera_world);

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


      // // Render normals
      // normalFrameBuffer.Bind();
      // glPushAttrib(GL_VIEWPORT_BIT);
      // glViewport(0, 0, width, height);
      // glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      // glEnable(GL_CULL_FACE);

      // ptexMesh.Render(s_cam, &normalShaderProgram);

      // glDisable(GL_CULL_FACE);

      // glPopAttrib(); // GL_VIEWPORT_BIT
      // normalFrameBuffer.Unbind();

      // normalTexture.Download(normalImage.data(), GL_RGB, GL_FLOAT);

      // saveData("normals.csv", normalImage, std::string(filename));

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

      // Move the camera
      T_camera_world = T_camera_world * T_new_old_inv;

      s_cam.SetModelViewMatrix(T_camera_world);
    }
    std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;
  }
  return 0;
}
