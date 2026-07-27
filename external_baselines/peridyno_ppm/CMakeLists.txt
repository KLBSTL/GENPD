if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY
        Modeling
        Peridynamics
        Multiphysics
        Volume
        UbiGUI)
    add_example(GenPD_PPM_OperatingPoint CodimensionalPD LIB_DEPENDENCY)
    set_target_properties(GenPD_PPM_OperatingPoint PROPERTIES OUTPUT_NAME GenPD_PPM_OperatingPoint_Rendered)
endif()
