
macro(move_release_to_src target_name)
    if(WIN32)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_custom_command(
            TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
                ${CMAKE_CURRENT_BINARY_DIR}/Debug/${target_name}.exe
                ${CMAKE_SOURCE_DIR}/${target_name}.exe
        )
    else()
        add_custom_command(
            TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
                ${CMAKE_CURRENT_BINARY_DIR}/Release/${target_name}.exe
                ${CMAKE_SOURCE_DIR}/${target_name}.exe
        )
    endif()
    else()
    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            ${CMAKE_CURRENT_BINARY_DIR}/${target_name}        
            ${CMAKE_SOURCE_DIR}/${target_name}    
    )
    endif()
endmacro()