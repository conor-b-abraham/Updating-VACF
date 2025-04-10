println("""────────────────────────────────────────────────────────────────────────────────────
        █▀▀█ █▀▀█ █   █ █▀▀▀ █▀▀█   █▀▀█ █▀▀█ █▀▀▀ █▀▀█ ▀▀█▀▀ █▀▀█ █  █ █▀▄▀█
        █▄▄█ █  █ █ █ █ █▀▀▀ █▄▄▀   ▀▀▄▄ █▄▄█ █▀▀▀ █      █   █▄▄▀ █  █ █ █ █
        █    █▄▄█ █▄▀▄█ █▄▄▄ █  █   █▄▄█ █    █▄▄▄ █▄▄█   █   █  █ ▀▄▄▀ █   █
────────────────────────────────────────────────────────────────────────────────────""")
# ------------------------------------------------------------------------------
# PACKAGES
# ------------------------------------------------------------------------------
using NPZ
using FFTW
using ArgParse
using Interpolations
# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
function init()
    arguments = ArgParseSettings()
    arguments.description = "This program computes the power (vibrational) spectrum for "*
                            "vacfs computed by updating_vacf.jl. Optionally, spline "*
                            "interpolation can be used to fit each vacf prior to performing "*
                            "the fourier transform to reduce noise in the spectrum. "
    arguments.epilog = "OUTPUT INFORMATION: "*
                       "The output files will be named the same as the input files, but "*
                       "'vacfs' will be replaced with 'spectrums' (e.g. the spectrums for "*
                       "the vacfs in 'example_vacfs.npz' will be saved to 'example_spectrums.npz'). "

    @add_arg_table arguments begin
        "--vacf_files", "-v"
            help = "One or more .npz file(s) written by updating_vacf.jl"
            arg_type = String
            nargs = '+'
        "--interpolate", "-i"
            help = "Perform spline interpolation of VACF(s) prior to computing the power spectrum."
            action = :store_true
    end

    parsed_args = parse_args(arguments)
    vacf_files = parsed_args["vacf_files"]
    interpolate_it = parsed_args["interpolate"]
    spectrum_files = Array{String, 1}(undef, length(vacf_files))
    error = false
    for (i, f) in enumerate(vacf_files)
        if '/' in f
            dir = f[1:findlast("/", f)[end]]
            file = f[findlast("/", f)[end]+1:end]
        else
            dir = ""
            file = f
        end
        if file[end-3:end] != ".npz" || !occursin("vacfs", file)
            println("\nERROR: The file, $file, is not a valid input file. All input files should come from updating_vacf.jl.")
            error = true
        end
        if (dir != "")
            if !isdir(dir)
                println("\nERROR: The directory $dir does not exist.")
                error = true
            end
        end
        if !isfile(f)
            println("\nERROR: The file $f does not exist")
            error = true
        end
        if !error
            spectrum_files[i] = dir*replace(file, "vacfs"=>"spectrums")
        end
    end
    if error
        exit(86)
    end

    println("IO DETAILS:")
    for (v, s) in zip(vacf_files, spectrum_files)
        println("$v 🠒 $s")
    end
    println("────────────────────────────────────────────────────────────────────────────────────")
    return vacf_files, spectrum_files, interpolate_it
end

function power_spectrum(vacf::Vector{Float64}, ts::Float64)
    # Compute the power spectrum from a velocity autocorrelation function
    # i.e. find the absolute value of the fast fourier transform of the VACF
    # for this, we use the rfft set of functions from FFTW to not calculate
    # values for negative frequencies.
    sfreq = (1/ts)/0.0299792458 # sampling frequency in wavenumbers (1/cm)
    spectrum = abs.(rfft(vacf)) # Real fast fourier transform of VACF (ignore negative values)
    frequencies = rfftfreq(length(vacf), sfreq) # frequencies
    return frequencies, spectrum
end

function interpolate_vacf(times::Vector{Float64}, vacf::Vector{Float64})
    x = LinRange(minimum(times), maximum(times), length(times))
    spline = CubicSplineInterpolation(x, vacf)
    new_times = LinRange(minimum(times), maximum(times), round(Int, length(times)/10))
    new_vacf = spline(new_times)
    return new_times, new_vacf
end

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
function main()
    vacf_files, spectrum_files, interpolate_it = init()
    nfiles = length(vacf_files)
    for (i, (v, s)) in enumerate(zip(vacf_files, spectrum_files))
        println("VACFs from: $v ($i/$nfiles)")
        vacfs = npzread(v)
        ts = vacfs["times"][2]-vacfs["times"][1]
        println("Timestep: $ts ps")

        if interpolate_it
            println("Interpolating total vacf")
            _, vacfs["total"] = interpolate_vacf(vacfs["times"], vacfs["total"])
            println("Interpolating ∥, ∥ vacf")
            _, vacfs["ll_ll"] = interpolate_vacf(vacfs["times"], vacfs["ll_ll"])
            println("Interpolating ⟂, ⟂ vacf")
            vacfs["times"], vacfs["T_T"] = interpolate_vacf(vacfs["times"], vacfs["T_T"])
            ts = vacfs["times"][2]-vacfs["times"][1]
            println("Timestep updated to: $ts ps")
        end

        println("Computing Total Power Spectrum")
        frequencies, spectrum_total = power_spectrum(vacfs["total"], ts)
        println("Computing ∥, ∥ Power Spectrum")
        _, spectrum_ll_ll = power_spectrum(vacfs["ll_ll"], ts)
        println("Computing ⟂, ⟂ Power Spectrum")
        _, spectrum_T_T = power_spectrum(vacfs["T_T"], ts)

        npzwrite(s, frequencies=frequencies, total=spectrum_total, ll_ll=spectrum_ll_ll,
                 T_T=spectrum_T_T)

        println("Power Spectrums saved to: $s\n")
    end
    println("────────────────────────────────────────────────────────────────────────────────────")
    println("The spectrum file(s) contain(s) a dictionary with the following items:\n"*
            "\"frequencies\" 🠒 The frequencies for all power spectrums (in 1/cm)\n"*
            "      \"total\" 🠒 The total power spectrum\n"*
            "      \"ll_ll\" 🠒 The ∥, ∥ power spectrum\n"*
            "        \"T_T\" 🠒 The ⟂, ⟂ power spectrum\n")
end
main()
