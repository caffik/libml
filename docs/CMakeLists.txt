message(STATUS "libml: Configuring and building documentation...")

find_package(Doxygen)

if (DOXYGEN_FOUND)
    message(STATUS "Generating Doxygen documentation...")
    execute_process(
        COMMAND ${DOXYGEN_EXECUTABLE} ${Doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_QUIET
    )
else()
    message(STATUS "Doxygen not found. Skipping documentation generation.")
endif()

find_package(LATEX COMPONENTS PDFLATEX)

if (LATEX_FOUND)
    message(STATUS "Generating PDF documentation...")
    execute_process(
        COMMAND ${PDFLATEX_COMPILER} refman.tex
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/latex
        OUTPUT_QUIET
    )
else ()
    message(STATUS "PDFLatex not found. Skipping PDF documentation generation.")
endif ()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/latex/refman.pdf
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
