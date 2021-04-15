#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragCameraPos;
layout(location = 3) in vec3 fragVertexWorldPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

bool JudgeIsNan(float val)
{
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
}

void main() {
    if(isnan(fragTexCoord.x) == true || isnan( fragTexCoord.y) == true )
    {
        outColor = vec4(fragColor, 1.0);
        
    }
    else
    {
        
        outColor = texture(texSampler, fragTexCoord);

        // add blur for remote textures
        float length = length(fragCameraPos - fragVertexWorldPos);
        float begin_blur_dist = 20;
        if(length > begin_blur_dist)
        {   
            outColor += 0.1 * vec4(1, 1, 1 ,1)* log(length - begin_blur_dist + 1);
        }
        
    }
    
    // outColor = vec4(fragVertexWorldPos.x, fragVertexWorldPos.y, fragVertexWorldPos.z, 1.0);
    // outColor = vec4(fragCameraPos.x, fragCameraPos.y,  fragCameraPos.z, 1.0) / 5;

    

    // outColor *= length / 10;
    // if(length > 1.0)
    // {
    //     outColor =  vec4(diff, 1.0);
    // }
    // outColor = texture(texSampler, fragTexCoord);
    // float camera_vertex_dist = (fragCameraPos - fragVertexWorldPos).length();
    // outColor *= fragVertexWorldPos.length() / 5;
    // outColor =  * camera_vertex_dist;
    // if(camera_vertex_dist < 4.0)
    // {
    //     outColor = vec4(1.0, 0.0, 0.0, 1.0);
    // }
}