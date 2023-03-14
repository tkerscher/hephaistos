#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <hephaistos/image.hpp>
#include <hephaistos/program.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

extern nb::class_<hp::Command> commandClass;

//array shape of image buffer: [width, height, 4]
using image_shape = nb::shape<nb::any, nb::any, 4>;
using image_array = nb::ndarray<nb::numpy, uint8_t, image_shape>;

void registerImageModule(nb::module_& m) {
    nb::enum_<hp::ImageFormat>(m, "ImageFormat")
        .value("R8G8B8A8_UNORM",      hp::ImageFormat::R8G8B8A8_UNORM)
        .value("R8G8B8A8_SNORM",      hp::ImageFormat::R8G8B8A8_SNORM)
        .value("R8G8B8A8_UINT",       hp::ImageFormat::R8G8B8A8_UINT)
        .value("R8G8B8A8_SINT",       hp::ImageFormat::R8G8B8A8_SINT)
        .value("R16G16B16A16_UINT",   hp::ImageFormat::R16G16B16A16_UINT)
        .value("R16G16B16A16_SINT",   hp::ImageFormat::R16G16B16A16_SINT)
        .value("R32_UINT",            hp::ImageFormat::R32_UINT)
        .value("R32_SINT",            hp::ImageFormat::R32_SINT)
        .value("R32_SFLOAT",          hp::ImageFormat::R32_SFLOAT)
        .value("R32G32_UINT",         hp::ImageFormat::R32G32_UINT)
        .value("R32G32_SINT",         hp::ImageFormat::R32G32_SINT)
        .value("R32G32_SFLOAT",       hp::ImageFormat::R32G32_SFLOAT)
        .value("R32G32B32A32_UINT",   hp::ImageFormat::R32G32B32A32_UINT)
        .value("R32G32B32A32_SINT",   hp::ImageFormat::R32G32B32A32_SINT)
        .value("R32G32B32A32_SFLOAT", hp::ImageFormat::R32G32B32A32_SFLOAT)
        .export_values();
    m.def("getElementSize", &hp::getElementSize, "format"_a);

    nb::class_<hp::Image>(m, "Image")
        .def("__init__",
            [](hp::Image* image, hp::ImageFormat format,
                uint32_t width, uint32_t height, uint32_t depth)
            {
                new (image) hp::Image(getCurrentContext(), format, width, height, depth);
            }, "format"_a, "width"_a, "height"_a = 1, "depth"_a = 1)
        .def_prop_ro("width", [](const hp::Image& img) -> uint32_t { return img.getWidth(); })
        .def_prop_ro("height", [](const hp::Image& img) -> uint32_t { return img.getHeight(); })
        .def_prop_ro("depth", [](const hp::Image& img) -> uint32_t { return img.getDepth(); })
        .def_prop_ro("format", [](const hp::Image& img) -> hp::ImageFormat { return img.getFormat(); })
        .def_prop_ro("size_bytes", [](const hp::Image& img) -> uint64_t { return img.size_bytes(); })
        .def("bindParameter", [](const hp::Image& img, hp::Program& p, uint32_t b) { img.bindParameter(p.getBinding(b)); });
    
    nb::class_<hp::ImageBuffer, hp::Buffer<std::byte>>(m, "ImageBuffer")
        .def("__init__",
            [](hp::ImageBuffer* ib, uint32_t width, uint32_t height) {
                new (ib) hp::ImageBuffer(getCurrentContext(), width, height);
            }, "width"_a, "height"_a)
        .def_prop_ro("width", [](const hp::ImageBuffer& ib) { return ib.getWidth(); })
        .def_prop_ro("height", [](const hp::ImageBuffer& ib) { return ib.getHeight(); })
        //TODO: This is somehow broken, but I don't get a proper error...
        //.def("createImage", &hp::ImageBuffer::createImage, "copy"_a = true)
        //.def("createImage", [](const hp::ImageBuffer& ib, bool copy) -> hp::Image { return ib.createImage(copy); }, "copy"_a = true)
        //.def("createImage", [](const hp::ImageBuffer& ib) -> hp::Image { return hp::Image(getCurrentContext(), hp::ImageFormat::R8G8B8A8_UNORM, ib.getWidth(), ib.getHeight()); })
        .def("save", &hp::ImageBuffer::save, "filename"_a)
        .def_static("loadFile", [](const char* filename) -> hp::ImageBuffer {
                return hp::ImageBuffer::load(getCurrentContext(), filename);
            }, "filename"_a)
        .def_static("loadMemory", [](nb::bytes data) -> hp::ImageBuffer {
                return hp::ImageBuffer::load(
                    getCurrentContext(),
                    std::span<const std::byte>{ reinterpret_cast<const std::byte*>(data.c_str()), data.size() });
            }, "data"_a)
        .def("numpy", [](const hp::ImageBuffer& buffer) -> image_array {
                size_t shape[3] = { buffer.getHeight(), buffer.getWidth(), 4 };
                return image_array(static_cast<void*>(buffer.getMemory().data()), 3, shape);
            });
    
    nb::class_<hp::RetrieveImageCommand>(m, "RetrieveImageCommand", commandClass)
        .def(nb::init<const hp::Image&, const hp::Buffer<std::byte>&>());
    m.def("retrieveImage", &hp::retrieveImage, "src"_a, "dst"_a,
        "Creates a command for copying the image back into the given buffer.");
    nb::class_<hp::UpdateImageCommand>(m, "UpdateImageCommand", commandClass)
        .def(nb::init<const hp::Buffer<std::byte>&, const hp::Image&>());
    m.def("updateImage", &hp::updateImage, "src"_a, "dst"_a,
        "Creates a command for copying data from the given buffer into the image.");
}
