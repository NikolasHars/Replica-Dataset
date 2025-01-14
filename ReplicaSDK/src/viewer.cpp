// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <PTexLib.h>

#include <pangolin/display/display.h>
#include <pangolin/display/widgets/widgets.h>

#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>

#include <yaml-cpp/yaml.h>

#include <map>
#include <vector>

#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <cmath>

#include "GLCheck.h"
#include "MirrorRenderer.h"

using namespace std;
using namespace Eigen;

std::string YAML_FILE = "/cluster/project/infk/courses/252-0579-00L/group04/processed_data/conf_viewer.yaml";
auto CONFIG = YAML::LoadFile(YAML_FILE);

void saveData(std::string fileName, MatrixXd  matrix, std::string img_name)
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

  // Output folder and pose file
  std::string name = scene_name;
  std::string pose_file_name = name + "_poses.csv";

  int number = 0;

  const int uiWidth = 180;

  // Setup OpenGL Display (based on GLUT)
  pangolin::CreateWindowAndBind("ReplicaViewer", uiWidth + width, height);

  if (glewInit() != GLEW_OK) {
    pango_print_error("Unable to initialize GLEW.");
  }

  if(!checkGLVersion()) {
    return 1;
  }

  // Setup default OpenGL parameters
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  const GLenum frontFace = GL_CW;
  glFrontFace(frontFace);
  glLineWidth(1.0f);

  // Tell the base view to arrange its children equally
  if (uiWidth != 0) {
    pangolin::CreatePanel("ui").SetBounds(0, 1.0f, 0, pangolin::Attach::Pix(uiWidth));
  }

  pangolin::View& container =
      pangolin::CreateDisplay().SetBounds(0, 1.0f, pangolin::Attach::Pix(uiWidth), 1.0f);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_TopLeft(
          width,
          height,
          width / 2.0f,
          width / 2.0f,
          (width - 1.0f) / 2.0f,
          (height - 1.0f) / 2.0f,
          0.1f,
          100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  pangolin::Handler3D s_handler(s_cam);

  pangolin::View& meshView = pangolin::Display("MeshView")
                                 .SetBounds(0, 1.0f, 0, 1.0f, (double)width / (double)height)
                                 .SetHandler(&s_handler);

  container.AddDisplay(meshView);

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

  pangolin::Var<float> exposure("ui.Exposure", 0.01, 0.0f, 0.1f);
  pangolin::Var<float> gamma("ui.Gamma", ptexMesh.Gamma(), 1.0f, 3.0f);
  pangolin::Var<float> saturation("ui.Saturation", ptexMesh.Saturation(), 0.0f, 2.0f);
  pangolin::Var<float> depthScale("ui.Depth_scale", 0.1f, 0.0f, 1.0f);

  pangolin::Var<bool> wireframe("ui.Wireframe", false, true);
  pangolin::Var<bool> drawBackfaces("ui.Draw_backfaces", false, true);
  pangolin::Var<bool> drawMirrors("ui.Draw_mirrors", true, true);
  pangolin::Var<bool> drawDepth("ui.Draw_depth", false, true);

  pangolin::Var<bool> togglePose("ui.Toggle_pose", false, true);

  ptexMesh.SetExposure(exposure);


  Eigen::Matrix4d T_camera_world;
  while (!pangolin::ShouldQuit()) {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    if (exposure.GuiChanged()) {
      ptexMesh.SetExposure(exposure);
    }

    if (gamma.GuiChanged()) {
      ptexMesh.SetGamma(gamma);
    }

    if (saturation.GuiChanged()) {
      ptexMesh.SetSaturation(saturation);
    }

    if (meshView.IsShown()) {
      meshView.Activate(s_cam);

      if (drawBackfaces) {
        glDisable(GL_CULL_FACE);
      } else {
        glEnable(GL_CULL_FACE);
      }

      if (wireframe) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
        ptexMesh.Render(s_cam);
        glDisable(GL_POLYGON_OFFSET_FILL);
        // render wireframe on top
        ptexMesh.RenderWireframe(s_cam);
      } else if (drawDepth) {
        ptexMesh.RenderDepth(s_cam, depthScale);
      } else {
        ptexMesh.Render(s_cam);
      }

      if (togglePose) {
        T_camera_world = s_cam.GetModelViewMatrix();
        saveData(pose_file_name, T_camera_world, name + std::to_string(number));
        togglePose = false;
        number++;
      }

      glDisable(GL_CULL_FACE);

      if (drawMirrors) {
        for (size_t i = 0; i < mirrors.size(); i++) {
          MirrorSurface& mirror = mirrors[i];
          // capture reflections
          mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace, drawDepth, depthScale);

          // render mirror
          mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam, drawDepth);
        }
      }
    }

    pangolin::FinishFrame();
  }

  return 0;
}
