#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

bool JudgeIsNan(float val)
{
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
}

void main() {
    if(isnan( fragTexCoord.x) == true || isnan( fragTexCoord.y) == true )
    {
        outColor = vec4(fragColor, 1.0);
        
    }
    else
    {
        // outColor = vec4(0.1, 0.1, 0.1, 1.0);
        // outColor = vec4(fragTexCoord, 0.1, 1.0);
        outColor = texture(texSampler, fragTexCoord);
    }
    
    // outColor = texture(texSampler, fragTexCoord);
    
}