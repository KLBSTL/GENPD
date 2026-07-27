#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <SceneGraph.h>
#include <Timer.h>

#include <GlfwApp.h>
#include <GlfwRenderWindow.h>
#include <GLSurfaceVisualModule.h>
#include <Node/GLSurfaceVisualNode.h>

#include <BasicShapes/SphereModel.h>
#include <Multiphysics/VolumeBoundary.h>
#include <Peridynamics/CodimensionalPD.h>
#include <Peridynamics/Module/CoSemiImplicitHyperelasticitySolver.h>
#include <Peridynamics/Module/FixedPoints.h>
#include <Topology/TriangleSet.h>
#include <Volume/BasicShapeToVolume.h>

using namespace dyno;

namespace
{
constexpr float kSceneScale = 0.1f;
constexpr float kFrameDt = 0.0333f;

struct Options
{
    std::string scene = "hanging";
    std::string output_dir = "results/peridyno-ppm-operating-point";
    std::string peridyno_commit = "unknown";
    int width = 64;
    int height = 64;
    int warmup = 8;
    int frames = 30;
    int solver_iterations = 10;
    int render_width = 1600;
    int render_height = 900;
    float dt = 0.001f;
    bool headless = false;
    std::string capture_output;
};

struct FrameRecord
{
    int frame = 0;
    float host_ms = 0.0f;
    float simulation_gpu_ms = 0.0f;
    bool rendered = false;
};

struct SceneObjects
{
    std::shared_ptr<SceneGraph> graph;
    std::shared_ptr<CodimensionalPD<DataType3f>> cloth;
    std::shared_ptr<SphereModel<DataType3f>> sphere;
    std::shared_ptr<GLSurfaceVisualNode<DataType3f>> sphere_visualizer;
    std::shared_ptr<BasicShapeToVolume<DataType3f>> sphere_volume;
    int substeps_per_frame = 0;
};

void PrintUsage()
{
    std::cout
        << "Usage: GenPD_PPM_OperatingPoint [options]\n"
        << "  --scene hanging|moving-sphere\n"
        << "  --width N --height N\n"
        << "  --warmup N --frames N\n"
        << "  --solver-iterations N --dt seconds\n"
        << "  --render-width W --render-height H\n"
        << "  --output-dir PATH --peridyno-commit SHA\n"
        << "  --capture-output PATH (written after timing; rendered mode only)\n"
        << "  --headless (diagnostic only; skips GLFW initialization)\n";
}

bool ParseInt(const char* text, int& value)
{
    try
    {
        value = std::stoi(text);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ParseFloat(const char* text, float& value)
{
    try
    {
        value = std::stof(text);
        return std::isfinite(value);
    }
    catch (...)
    {
        return false;
    }
}

bool ParseOptions(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help")
        {
            PrintUsage();
            return false;
        }

        if (arg == "--headless")
        {
            options.headless = true;
            continue;
        }

        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for " << arg << "\n";
            return false;
        }

        const char* value = argv[++i];
        bool parsed = true;
        if (arg == "--scene") options.scene = value;
        else if (arg == "--width") parsed = ParseInt(value, options.width);
        else if (arg == "--height") parsed = ParseInt(value, options.height);
        else if (arg == "--warmup") parsed = ParseInt(value, options.warmup);
        else if (arg == "--frames") parsed = ParseInt(value, options.frames);
        else if (arg == "--solver-iterations") parsed = ParseInt(value, options.solver_iterations);
        else if (arg == "--render-width") parsed = ParseInt(value, options.render_width);
        else if (arg == "--render-height") parsed = ParseInt(value, options.render_height);
        else if (arg == "--dt") parsed = ParseFloat(value, options.dt);
        else if (arg == "--output-dir") options.output_dir = value;
        else if (arg == "--peridyno-commit") options.peridyno_commit = value;
        else if (arg == "--capture-output") options.capture_output = value;
        else
        {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }

        if (!parsed)
        {
            std::cerr << "Invalid value for " << arg << ": " << value << "\n";
            return false;
        }
    }

    if ((options.scene != "hanging" && options.scene != "moving-sphere") ||
        options.width < 2 || options.height < 2 || options.warmup < 0 ||
        options.frames < 1 || options.solver_iterations < 1 || options.render_width < 1 ||
        options.render_height < 1 || options.dt <= 0.0f)
    {
        std::cerr << "Invalid operating-point configuration.\n";
        return false;
    }

    return true;
}

void MakeGenPDGrid(const Options& options,
                   std::vector<Vec3f>& points,
                   std::vector<Topology::Triangle>& triangles)
{
    // This reproduces GenPD's cloth-grid parameterization after a uniform 0.1 scale.
    const Vec3f corner0(-5.0f * kSceneScale, 8.0f * kSceneScale, -5.0f * kSceneScale);
    const Vec3f corner1(5.0f * kSceneScale, -1.85f * kSceneScale, -3.26f * kSceneScale);
    const Vec3f delta_x((corner1.x - corner0.x) / static_cast<float>(options.width - 1), 0.0f, 0.0f);
    const Vec3f delta_yz(0.0f,
                          (corner1.y - corner0.y) / static_cast<float>(options.height - 1),
                          (corner1.z - corner0.z) / static_cast<float>(options.height - 1));

    points.reserve(static_cast<size_t>(options.width) * options.height);
    for (int x = 0; x < options.width; ++x)
    {
        for (int y = 0; y < options.height; ++y)
        {
            points.push_back(corner0 + static_cast<float>(x) * delta_x + static_cast<float>(y) * delta_yz);
        }
    }

    triangles.reserve(static_cast<size_t>(options.width - 1) * (options.height - 1) * 2);
    bool row_flip = false;
    bool column_flip = false;
    for (int x = 0; x < options.width - 1; ++x)
    {
        for (int y = 0; y < options.height - 1; ++y)
        {
            const int p00 = options.height * x + y;
            const int p01 = p00 + 1;
            const int p10 = options.height * (x + 1) + y;
            const int p11 = p10 + 1;
            const bool diagonal_flip = row_flip ^ column_flip;

            triangles.emplace_back(p00, p01, diagonal_flip ? p11 : p10);
            triangles.emplace_back(p11, p10, diagonal_flip ? p00 : p01);
            row_flip = !row_flip;
        }
        column_flip = !column_flip;
        row_flip = false;
    }
}

SceneObjects CreateScene(const Options& options)
{
    SceneObjects objects;
    objects.graph = std::make_shared<SceneGraph>();
    objects.graph->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
    objects.graph->setVerboseMode(false);

    std::vector<Vec3f> points;
    std::vector<Topology::Triangle> triangles;
    MakeGenPDGrid(options, points, triangles);

    if (options.scene == "moving-sphere")
    {
        objects.sphere = objects.graph->addNode(std::make_shared<SphereModel<DataType3f>>());
        objects.sphere->varLocation()->setValue(Vec3f(0.0f, 10.0f * kSceneScale, -2.5f * kSceneScale));
        objects.sphere->varRadius()->setValue(2.0f * kSceneScale);
        objects.sphere->varLatitude()->setValue(16);
        objects.sphere->varLongitude()->setValue(16);
        objects.sphere->setVisible(true);

        objects.sphere_volume = objects.graph->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
        objects.sphere_volume->varGridSpacing()->setValue(0.02f);
        objects.sphere->connect(objects.sphere_volume->importShape());
    }

    objects.cloth = objects.graph->addNode(std::make_shared<CodimensionalPD<DataType3f>>());
    objects.cloth->setVisible(true);
    auto mesh = objects.cloth->stateTriangleSet()->getDataPtr();
    mesh->setPoints(points);
    mesh->setTriangles(triangles);
    mesh->update();

    const float spacing_x = 1.0f / static_cast<float>(options.width - 1);
    const float spacing_y = std::sqrt(0.985f * 0.985f + 0.174f * 0.174f) /
                            static_cast<float>(options.height - 1);
    objects.cloth->stateHorizon()->setValue(2.2f * (std::max)(spacing_x, spacing_y));
    objects.cloth->setDt(options.dt);

    auto cloth_renderer = std::make_shared<GLSurfaceVisualModule>();
    cloth_renderer->varBaseColor()->setValue(Color(0.82f, 0.24f, 0.18f));
    objects.cloth->stateTriangleSet()->connect(cloth_renderer->inTriangleSet());
    objects.cloth->graphicsPipeline()->pushModule(cloth_renderer);

    if (objects.sphere)
    {
        // Follow PeriDyno's official cross-node visualization pattern for the moving sphere.
        objects.sphere_visualizer = objects.graph->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
        objects.sphere->stateTriangleSet()->connect(objects.sphere_visualizer->inTriangleSet());
        objects.sphere_visualizer->varColor()->setValue(Vec3f(0.16f, 0.40f, 0.78f));
        objects.sphere_visualizer->setVisible(true);
    }

    auto solver = objects.cloth->animationPipeline()->findFirstModule<CoSemiImplicitHyperelasticitySolver<DataType3f>>();
    solver->setGrad_res_eps(0.0f);
    solver->varIterationNumber()->setValue(options.solver_iterations);
    solver->setS(0.1f);
    solver->setXi(0.15f);
    solver->setE(500.0f);
    solver->setK_bend(0.0005f);
    solver->setSelfContact(false);

    auto fixed_points = std::make_shared<FixedPoints<DataType3f>>();
    objects.cloth->statePosition()->connect(fixed_points->inPosition());
    objects.cloth->stateVelocity()->connect(fixed_points->inVelocity());
    // PeriDyno initializes these optional serialized fields even when points are added directly.
    fixed_points->FixedIds.resize(0);
    fixed_points->FixedPos.resize(0);
    fixed_points->addFixedPoint(0, points.front());
    fixed_points->addFixedPoint(options.height * (options.width - 1), points[options.height * (options.width - 1)]);
    objects.cloth->animationPipeline()->pushModule(fixed_points);

    if (objects.sphere_volume)
    {
        auto boundary = objects.graph->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
        objects.sphere_volume->connect(boundary->importVolumes());
        objects.cloth->connect(boundary->importTriangularSystems());
    }

    objects.substeps_per_frame = (std::max)(1, static_cast<int>(std::round(kFrameDt / options.dt)));
    return objects;
}

void UpdateMovingSphere(SceneObjects& objects, const Options& options, int substep)
{
    if (!objects.sphere || !objects.sphere_volume)
    {
        return;
    }

    const float elapsed_seconds = static_cast<float>(substep) * options.dt;
    const float y = (10.0f - 1.2f * elapsed_seconds) * kSceneScale;
    objects.sphere->varLocation()->setValue(Vec3f(0.0f, y, -2.5f * kSceneScale));
    // Refresh the visual triangle mesh before rebuilding the matching collision SDF.
    objects.sphere->reset();
    // BasicShapeToVolume rebuilds its SDF on reset; this is the explicit moving-obstacle adapter.
    objects.sphere_volume->reset();
}

FrameRecord AdvanceFrame(SceneObjects& objects,
                         const Options& options,
                         int frame,
                         GlfwRenderWindow* render_window)
{
    GTimer gpu_timer;
    CTimer host_timer;
    gpu_timer.start();
    host_timer.start();

    const int first_substep = frame * objects.substeps_per_frame;
    for (int i = 0; i < objects.substeps_per_frame; ++i)
    {
        UpdateMovingSphere(objects, options, first_substep + i);
        objects.graph->advance(options.dt);
    }

    gpu_timer.stop();
    bool rendered = false;
    if (render_window)
    {
        rendered = render_window->renderOneFrame();
    }
    host_timer.stop();
    return FrameRecord{ frame, static_cast<float>(host_timer.getElapsedTime()), gpu_timer.getElapsedTime(), rendered };
}

bool IsFiniteFinalState(const std::shared_ptr<CodimensionalPD<DataType3f>>& cloth, float& max_abs_position)
{
    CArray<Vec3f> host_positions;
    host_positions.assign(cloth->statePosition()->getData());

    max_abs_position = 0.0f;
    for (uint i = 0; i < host_positions.size(); ++i)
    {
        const Vec3f& position = host_positions[i];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
        {
            return false;
        }
        const float max_coordinate = (std::max)(std::abs(position.x),
                                                (std::max)(std::abs(position.y), std::abs(position.z)));
        max_abs_position = (std::max)(max_abs_position, max_coordinate);
    }
    return true;
}

void WriteResults(const Options& options,
                  const SceneObjects& objects,
                  const std::vector<FrameRecord>& records,
                  bool finite,
                  float max_abs_position)
{
    const std::filesystem::path output_dir(options.output_dir);
    std::filesystem::create_directories(output_dir);

    std::ofstream csv(output_dir / "frame_profile.csv");
    csv << "frame,scene,vertex_count,triangle_count,dt,substeps_per_frame,solver_iterations,frame_host_ms,simulation_gpu_ms,rendered,moving_sdf_refresh\n";
    csv << std::fixed << std::setprecision(6);
    const int vertex_count = options.width * options.height;
    const int triangle_count = 2 * (options.width - 1) * (options.height - 1);
    const int moving_sdf_refresh = options.scene == "moving-sphere" ? 1 : 0;
    int sphere_render_vertex_count = 0;
    int sphere_render_triangle_count = 0;
    Vec3f sphere_render_min(0.0f);
    Vec3f sphere_render_max(0.0f);
    if (objects.sphere)
    {
        const auto sphere_mesh = objects.sphere->stateTriangleSet()->getDataPtr();
        sphere_render_vertex_count = static_cast<int>(sphere_mesh->getPoints().size());
        sphere_render_triangle_count = static_cast<int>(sphere_mesh->triangleIndices().size());
        CArray<Vec3f> host_sphere_positions;
        host_sphere_positions.assign(sphere_mesh->getPoints());
        if (host_sphere_positions.size() > 0)
        {
            sphere_render_min = host_sphere_positions[0];
            sphere_render_max = host_sphere_positions[0];
            for (uint i = 1; i < host_sphere_positions.size(); ++i)
            {
                const Vec3f& point = host_sphere_positions[i];
                sphere_render_min.x = (std::min)(sphere_render_min.x, point.x);
                sphere_render_min.y = (std::min)(sphere_render_min.y, point.y);
                sphere_render_min.z = (std::min)(sphere_render_min.z, point.z);
                sphere_render_max.x = (std::max)(sphere_render_max.x, point.x);
                sphere_render_max.y = (std::max)(sphere_render_max.y, point.y);
                sphere_render_max.z = (std::max)(sphere_render_max.z, point.z);
            }
        }
    }
    for (const FrameRecord& record : records)
    {
        csv << record.frame << ',' << options.scene << ',' << vertex_count << ',' << triangle_count << ','
            << options.dt << ',' << objects.substeps_per_frame << ',' << options.solver_iterations << ','
            << record.host_ms << ',' << record.simulation_gpu_ms << ','
            << (record.rendered ? 1 : 0) << ',' << moving_sdf_refresh << '\n';
    }

    std::ofstream metadata(output_dir / "run_metadata.json");
    metadata << "{\n"
             << "  \"baseline\": \"Projective Peridynamic Modeling of Hyperelastic Membranes With Contact (2024)\",\n"
             << "  \"peridyno_commit\": \"" << options.peridyno_commit << "\",\n"
             << "  \"comparison_scope\": \"same-hardware operating point; not an equal-model or equal-quality ranking\",\n"
             << "  \"scene\": \"" << options.scene << "\",\n"
             << "  \"vertex_count\": " << options.width * options.height << ",\n"
             << "  \"triangle_count\": " << 2 * (options.width - 1) * (options.height - 1) << ",\n"
             << "  \"warmup_frames\": " << options.warmup << ",\n"
             << "  \"measured_frames\": " << options.frames << ",\n"
             << "  \"dt_seconds\": " << options.dt << ",\n"
             << "  \"substeps_per_frame\": " << objects.substeps_per_frame << ",\n"
             << "  \"solver_iterations\": " << options.solver_iterations << ",\n"
             << "  \"rendered\": " << (options.headless ? "false" : "true") << ",\n"
             << "  \"render_width\": " << options.render_width << ",\n"
             << "  \"render_height\": " << options.render_height << ",\n"
             << "  \"timing_scope\": \"frame_host_ms includes PPM substeps plus graphics update, draw, and buffer swap; simulation_gpu_ms excludes the OpenGL draw\",\n"
             << "  \"self_contact\": false,\n"
             << "  \"moving_sdf_refresh\": " << (options.scene == "moving-sphere" ? "true" : "false") << ",\n"
             << "  \"sphere_render_vertex_count\": " << sphere_render_vertex_count << ",\n"
             << "  \"sphere_render_triangle_count\": " << sphere_render_triangle_count << ",\n"
             << "  \"sphere_render_min\": [" << sphere_render_min.x << ", " << sphere_render_min.y << ", " << sphere_render_min.z << "],\n"
             << "  \"sphere_render_max\": [" << sphere_render_max.x << ", " << sphere_render_max.y << ", " << sphere_render_max.z << "],\n"
             << "  \"final_state_finite\": " << (finite ? "true" : "false") << ",\n"
             << "  \"final_max_abs_position\": " << max_abs_position << "\n"
             << "}\n";
}
}

int main(int argc, char** argv)
{
    Options options;
    if (!ParseOptions(argc, argv, options))
    {
        return argc > 1 && std::string(argv[1]) == "--help" ? 0 : 1;
    }

    SceneObjects scene = CreateScene(options);
    std::unique_ptr<GlfwApp> app;
    GlfwRenderWindow* render_window = nullptr;
    if (!options.headless)
    {
        app = std::make_unique<GlfwApp>();
        app->setSceneGraph(scene.graph);
        app->initialize(options.render_width, options.render_height);
        render_window = dynamic_cast<GlfwRenderWindow*>(app->renderWindow());
        if (!render_window)
        {
            std::cerr << "PeriDyno GLFW render window was not initialized.\n";
            return 3;
        }
        render_window->turnOffVSync();
        render_window->getCamera()->setEyePos(Vec3f(1.45f, 1.05f, 1.60f));
        render_window->getCamera()->setTargetPos(Vec3f(0.0f, 0.28f, -0.35f));
    }
    scene.graph->reset();
    for (int frame = 0; frame < options.warmup; ++frame)
    {
        AdvanceFrame(scene, options, frame, render_window);
    }

    std::vector<FrameRecord> records;
    records.reserve(options.frames);
    for (int frame = 0; frame < options.frames; ++frame)
    {
        records.push_back(AdvanceFrame(scene, options, options.warmup + frame, render_window));
    }

    float max_abs_position = 0.0f;
    const bool finite = IsFiniteFinalState(scene.cloth, max_abs_position);
    WriteResults(options, scene, records, finite, max_abs_position);
    if (render_window && !options.capture_output.empty())
    {
        const std::filesystem::path capture_path(options.capture_output);
        if (capture_path.has_parent_path())
        {
            std::filesystem::create_directories(capture_path.parent_path());
        }
        render_window->captureFrame(capture_path.string());
    }
    if (render_window)
    {
        // The official infinite GLFW loop releases graphics resources before window teardown.
        render_window->getRenderEngine()->terminate();
        // PeriDyno's GLFW destructor can block after this bounded, non-interactive loop.
        // This process has already flushed all benchmark output and rendered output, so let
        // Windows reclaim the window object at process exit instead of timing teardown.
        app.release();
    }
    std::cout << "PPM operating point complete: finite=" << (finite ? "true" : "false")
              << ", max_abs_position=" << max_abs_position << "\n";
    return finite ? 0 : 2;
}
