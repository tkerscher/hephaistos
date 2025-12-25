#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT int hitValue;

layout(shaderRecordEXT) buffer SBTRecord {
    int sbt_value;
};

void main() {
    hitValue = sbt_value;
}
