// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>
#include <string>
#include <cmath>
#include<fstream>
#include <iostream>
#include<Eigen/Dense>
#include<fstream>

#include "GLCheck.h"
#include "MirrorRenderer.h"

using namespace std;
using namespace Eigen;

#define PI 3.14159265

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

int main(int argc, char* argv[]) {
  // ASSERT(argc == 3 || argc == 4, "Usage: ./ReplicaRenderer mesh.ply /path/to/atlases [mirrorFile]");
  ASSERT(argc == 11 || argc == 12, "Usage: ./ReplicaRenderer mesh.ply /path/to/atlases height width rad num_imgs name x y z [mirrorFile]");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  if (argc == 12) {
    surfaceFile = std::string(argv[11]);
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  std::size_t pos;
  std::string arg = argv[4];
  const int width = std::stoi(arg, &pos);

  arg = argv[3];
  const int height = std::stoi(arg, &pos);
  bool renderDepth = true;
  float depthScale = 65535.0f * 0.1f;

  // Movement
  arg = argv[5];
  const float rad = std::stof(arg, &pos);

    // Movement
  arg = argv[8];
  const float x = std::stof(arg, &pos);

  arg = argv[9];
  const float y = std::stof(arg, &pos);

  arg = argv[10];
  const float z = std::stof(arg, &pos);

  // Render some frames
  arg = argv[6];
  const int numFrames = std::stoi(arg, &pos);
  float float_num = static_cast<float>(numFrames);

  float factor = 2 * PI / float_num;

  float sin_wave_factor = 1.0;

  std::string name = std::string(argv[7]);
  const char* c_name = name.c_str();

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

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // Setup a camera
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          width,
          height,
          width / 2.0f,
          width / 2.0f,
          (width - 1.0f) / 2.0f,
          (height - 1.0f) / 2.0f,
          0.1f,
          100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  // Start at some origin
  Eigen::Matrix4d T_camera_world = s_cam.GetModelViewMatrix();

  Eigen::Matrix3d x_rot;
  x_rot << 1, 0, 0,
           0, 0.8660254, 0.5,
           0, -0.5, 0.8660254;
  T_camera_world.col(3) = T_camera_world.col(3) + Eigen::Vector4d(x, y, z, 0);
  T_camera_world.block(0,0,3,3) = T_camera_world.block(0,0,3,3) * x_rot;
  

  s_cam.SetModelViewMatrix(T_camera_world);

  // And move to the left
  Eigen::Matrix4d T_new_old = Eigen::Matrix4d::Identity();

  Eigen::Matrix2d z_rot;
  z_rot << cos(factor), sin(factor),
           -sin(factor), cos(factor);
  T_new_old.block(0, 0, 2, 2) = z_rot; //Eigen::Matrix2dEigen::Vector3d(move_x, move_y, move_z);
  Eigen::Matrix4d T_new_old_inv = T_new_old.inverse();

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

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

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

    glPopAttrib(); //GL_VIEWPORT_BIT
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
    snprintf(filename, 1000, "imgs/frame_%06d.jpg", i);

    saveData(pose_file_name, T_camera_world, std::string(filename));

    pangolin::SaveImage(
        image.UnsafeReinterpret<uint8_t>(),
        pangolin::PixelFormatFromString("RGB24"),
        std::string(filename));

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

      snprintf(filename, 1000, "labels/depth_%06d.png", i);
      pangolin::SaveImage(
          depthImageInt.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("GRAY16LE"),
          std::string(filename), true, 34.0f);
    }

    // Move the camera
    T_camera_world = T_camera_world * T_new_old_inv;
    T_camera_world.col(3) = T_camera_world.col(3) + Eigen::Vector4d(0, 0, rad*cos(sin_wave_factor), 0);
    sin_wave_factor += 1.0;

    s_cam.SetModelViewMatrix(T_camera_world);
  }
  std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;

  return 0;
}
