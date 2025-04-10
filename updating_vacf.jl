println("""────────────────────────────────────────────────────────────────────────────────
         █  █ █▀▀█ █▀▀▄ █▀▀█ ▀▀█▀▀ ▀█▀ █▄  █ █▀▀█   █   █ █▀▀█ █▀▀█ █▀▀▀
         █  █ █▄▄█ █  █ █▄▄█   █    █  █ █ █ █ ▄▄    █ █  █▄▄█ █    █▀▀▀
         ▀▄▄▀ █    █▄▄▀ █  █   █   ▄█▄ █  ▀█ █▄▄█    ▀▄▀  █  █ █▄▄█ █
────────────────────────────────────────────────────────────────────────────────""")
# ------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------
using ArgParse
using Distributed

function get_args()
    arguments = ArgParseSettings()
    arguments.description = "This program performs a velocity autocorrelation function "*
                            "on an updating atom group. The user must provide three files. "*
                            "First, a velocity file containing the velocities of all atoms "*
                            "that could appear in the updating atom group. Second, an atom group "*
                            "file containing a boolean array that provides information about which "*
                            "atoms are in the updating atom group at each frame. Third, "*
                            "axis file containing reference axes to use in calculating the parallel "*
                            "and perpendicular components of the vacf. Please note that this program "*
                            "will not normalize the vacfs."
    arguments.epilog = "Example of array sizes: if you have a 1000 frame trajectory, are performing the "*
                       "VACF on a subset of water molecules, your system has 3000 total water "*
                       "molecules, and you're working in 3 dimensions: the velocity file "*
                       "(--velocity_file/-v) should contain an array of shape (1000, 3000, 3), "*
                       "the atom group file (--atom_group_file/-g) should be of shape (1000, 3000), "*
                       "and the axis file (--axis_file, -a) would be of shape (1000, 3)."
    @add_arg_table arguments begin
        "--velocity_file", "-v"
            help = "A .npy file containing velocities for any atom that could appear in "*
                   "the updating atom group for all frames in a trajectory. The first dimension "*
                   "should correspond to the trajectory frames, second dimension should "*
                   "correspond to the number of atoms, and third dimension should be equal "*
                   "to 3. The time units can be "*
                   "specified using the --md_timestep/-t option (or will default to 0.002ps), but "*
                   "the distance units must be Angstroms."
            arg_type = String
        "--atom_group_file", "-g"
            help = "A .npy file containing a boolean array detailing the updating atom "*
                   "group of shape matching the first two dimensions of the velocity array. "*
                   "Multiple files can be imported using this option to create a combined atom group, "*
                   "but all arrays in those files must be of the same shape."
            arg_type = String
            nargs = '+'
        "--axis_file", "-a"
            help = "A .npy file containing reference axes for which to transform the velocities "*
                   "to get the acfs of the velocities parallel and perpendicular to these axes. "*
                   "This must have the same dimensions as the first and third dimensions of the "*
                   "velocities file. One axis file must be provided for every atom group file "*
                   "that is provided. "
            arg_type = String
            nargs = '+'
        "--maxdelt", "-m"
            help = "The fraction of frames to be spanned by the maximum Δt. Must be less than 1."
            default = 0.5
            arg_type = Float64
        "--output", "-o"
            help = "The name stem of the output files (do not include .npy). If nothing is used "*
                   "(default) and an axes file is not provided, then the vacfs will be saved to "*
                   "vacfs.npz. If 'example' is used, the results will be saved to example_vacfs.npz."
            default = ""
            arg_type = String
        "--md_timestep", "-t"
            help = "The timestep size from your md simulation (in ps). For example, if you saved "*
                   "velocities every 0.002ps then use '0.002', or if you saved velocities every "*
                   "100ps then use '100'."
            default = 0.002
            arg_type = Float64
        "--skipcheck", "-s"
            help = "Don't check validity of updating atom group. By default, this program will first "*
                   "check to make sure that the updating atom group is never empty for a given initial "*
                   "time and Δt. For large updating atom groups (e.g. bulk water outside some cutoff "*
                   "distance from a protein), this could take a very long time and might not be necessary. "*
                   "This option should only be used if you are confident that at least one atom will be "*
                   "present for each t_i and Δt."
            action = :store_true
        "--nprocs", "-n"
            help = "Number of processors to use. Use 0 to use all processors"
            default = 0
            arg_type = Int64
    end

    return parse_args(arguments)
end

# READ INPUT OPTIONS & INITIALIZE WORKERS
parsed_args = get_args()
np = parsed_args["nprocs"]
if np == 0
    np = length(Sys.cpu_info())
end
addprocs(np-1)
println("Using $np processors\n")


# ------------------------------------------------------------------------------
# LOAD OTHER PACKAGES
# -----------------------------------------------------------------------------
@everywhere begin
    using NPZ
    using SharedArrays
    using ProgressMeter
    using Statistics

# ------------------------------------------------------------------------------
# FUNCTIONS & STRUCTURES
# ------------------------------------------------------------------------------
    struct constants
        vel::SharedArray{Float64, 3} # Changed from Float32 (If mda analysisfromfunction was used for collection)
        ax::SharedArray{Float64, 3}
        ag::SharedArray{Bool, 3}
        nframes::Int64
        nsubag::Int64
        maxdelt::Int64
        output::String
        timestep::Float64
    end
end

function make_constants(vel::SharedArray{Float64, 3}, ax::SharedArray{Float64, 3},
                        ag::SharedArray{Bool, 3}, maxdelt::Float64, output::String,
                        mdts::Float64) # vel changed from Float32 (If analysisfromfunction was used for collection)
    vshape = size(vel)
    nframes = vshape[1]
    maxdelt = floor(Int, nframes*maxdelt)
    C = constants(vel, ax, ag, nframes, size(ag)[3], maxdelt, output, mdts)
    return C
end

function init(parsed_args)
    velocity_file = parsed_args["velocity_file"]
    atom_group_file = parsed_args["atom_group_file"]
    axis_file = parsed_args["axis_file"]
    maxdelt = parsed_args["maxdelt"]
    output = parsed_args["output"]
    mdts = parsed_args["md_timestep"]
    skipcheck = parsed_args["skipcheck"]

    # Adjust output as needed & Check to make sure directories exist
    error = false
    outdir = "PWD"
    if length(output) != 0
        if isdir(output) # check if user provided just a directory as namestem
            outdir = output
            outname = ""
        else
            outname = output
        end
        if '/' in output # check to make sure any directory in namestem exists
            outdir = output[1:findlast("/", output)[end]]
            if !isdir(outdir)
                println("\nERROR: The directory $outdir does not exist. Please provide valid output namestem with (-output/-o).")
                error = true
            end
            outname = output[findlast("/", output)[end]+1:end]
        end
        if outname != "" && !(outname[end] in ['.', '_', '-', '/'])
            outname = outname*"_" # add an underscore so file names look better
            output = output*"_"
        end
    else
        outname = ""
    end
    if error == true
        exit(86)
    end

    # Let user know what is going to happen
    println("INPUT/OUTPUT DETAILS")
    println("Reading velocities from: $velocity_file")
    agstring = join(atom_group_file, ' ')
    println("Reading atom group from: $agstring")
    axstring = join(axis_file, ' ')
    println("Reading axes from: $axstring")
    println("Results will be saved to directory: $outdir")
    if outname == ""
        println("Default output file names will be used")
    else
        println("The namestem '$outname' will be added to the beginning of each output file's name")
    end

    # CHECK TO MAKE SURE FILES EXIST
    if isfile(velocity_file) == false
        error = true
        println("\nERROR: The velocity file, $velocity_file, does not exist.")
    end
    for i in axis_file
        if isfile(i) == false
            error = true
            println("\nERROR: The axis file, $axis_file, does not exist.")
        end
    end
    for i in atom_group_file
        if isfile(i) == false
            error = true
            println("\nERROR: The atom group file, $i, does not exist.")
        end
    end
    aginputlen = length(atom_group_file)
    axinputlen = length(axis_file)
    if aginputlen != axinputlen
        println("\nERROR: The number of atom group files ($aginputlen) and axis files ($axinputlen) provided do not match.")
        error = true
    end
    if error
        exit(86)
    end

    # CHECK TO MAKE SURE FILES ARE USABLE
    VELOCITIES = npzread(velocity_file)
    vel_size = size(VELOCITIES)
    if vel_size[3] != 3
        println("ERROR: The velocity file, $velocity_file, does not contain 3D vectors.")
        error = true
    end

    AXES = cat(npzread(axis_file[1]), dims=3)
    axes_size = size(AXES)
    if axes_size[2] != 3
        println("ERROR: The axis file, ", axis_file[1], ", does not contain 3D vectors.")
        error = true
    end
    for i in axis_file[2:end]
        newfile = npzread(i)
        if size(newfile)[2] != 3
            println("ERROR: The axis file, $i, does not contain 3D vectors.")
            error = true
        end
        if size(newfile)[1:2] != axes_size[1:2]
            println("ERROR: The axis file, $i, is not of the same shape as the other(s).")
            error = true
        end
        if !error
            AXES = cat(AXES, newfile, dims=3)
        end
    end

    ATOM_GROUPS = cat(npzread(atom_group_file[1]), dims=3)
    ag_size = size(ATOM_GROUPS)
    for i in atom_group_file[2:end]
        newfile = npzread(i)
        if size(newfile)[1:2] != ag_size[1:2]
            println("ERROR: The atom group file, $i, is not of the same shape as the other(s).")
            error = true
        end
        if !error
            ATOM_GROUPS = cat(ATOM_GROUPS, newfile, dims=3)
        end
    end

    if error
        exit(86)
    end

    # CHECK TO MAKE SURE FILES ARE COMPATIBLE
    if vel_size[1] != ag_size[1]
        println("\nERROR: The number of trajectory frames (dim=1) included in the velocity file (",
                vel_size[1], ") does not match the number of trajectory frames included in ",
                "the atom group file (", ag_size[1],"). Please read help text and try again.")
        error = true
    end
    if vel_size[2] != ag_size[2]
        println("\nERROR: The number of atoms (dim=2) included in the velocity file (",
                vel_size[1], ") does not match the number of atoms included in ",
                "the atom group file (", ag_size[1],"). Please read help text and try again.")
        error = true
    end

    if vel_size[1] != axes_size[1]
        println("\nERROR: The number of frames (dim=1) included in the velocity file (",
                vel_size[1], ") does not match the number of frames included in ",
                "the axis file (", axes_size[1],"). Please read help text and try again.")
        error = true
    end
    if vel_size[3] != axes_size[2]
        println("\nERROR: The dimensionality of the axes (", axes_size[2], "D, dim=2) and the "*
        "velocities (", vel_size[3], "D, dim=3) don't match. Please read help text and try again.")
        error = true
    end

    if error
        exit(86)
    end

    # CHECK TO MAKE SURE OTHER OPTIONS ARE VALID
    if maxdelt >= 1
        println("\nERROR: Maximum Δt must be < 1.\n")
        exit(86)
    end

    # GET RID OF EXCESS ATOMS IN ATOM POOL (i.e. Only include atoms that appear in atom group at some point)
    println("\nATTEMPTING ATOM POOL REDUCTION")
    used_atoms = any.(eachslice(ATOM_GROUPS, dims=2))
    ATOM_GROUPS = ATOM_GROUPS[:, used_atoms, :]
    VELOCITIES = VELOCITIES[:, used_atoms, :]
    if ag_size[2] != size(ATOM_GROUPS)[2]
        println("Atom pool size reduced from ", ag_size[2], " atoms to ", size(ATOM_GROUPS)[2], " atoms.")
    else
        println("Atom pool could not be reduced.")
    end

    # CONVERT ARRAYS TO SHARED ARRAYS FOR MULTIPROCESSING & CREATE CONSTANTS OBJECT
    ATOM_GROUPS = convert(SharedArray, ATOM_GROUPS)
    VELOCITIES = convert(SharedArray, VELOCITIES)
    AXES = convert(SharedArray, AXES)
    C = make_constants(VELOCITIES, AXES, ATOM_GROUPS, maxdelt, output, mdts)

    # PRINT DETAILS
    println("\nOPERATION DETAILS")
    println("  Atom Pool Size: ", size(VELOCITIES)[2], " atoms")
    println("Number of Frames: ", C.nframes, " frames")
    println("      Maximum Δt: ", C.maxdelt, " frames")
    println("       Time-Step: ", C.timestep, " ps")
    println(" Number of Procs: ", nprocs(), " processors")
    if skipcheck
        println("NOTE: Updating AtomGroup will not be checked for validity.")
    end
    println("────────────────────────────────────────────────────────────────────────────────────")
    return C, skipcheck
end

function evaluate_atomgroup(C::constants)
    println("CHECKING UPDATING ATOM GROUP")
    ag_size = convert(SharedArray, zeros(C.maxdelt+1, C.nframes))
    ag_used = convert(SharedArray, zeros(Bool, C.maxdelt+1, C.nframes))
    @showprogress pmap(1:C.maxdelt) do dt # Changed from 0:C.maxdelt
        for t_i in 1:C.nframes-dt
            for sag in 1:C.nsubag
                ag_size[dt, t_i] += sum(C.ag[t_i+dt, :, sag].*C.ag[t_i, :, sag])
                ag_used[dt, t_i] = true
                if ag_size[dt, t_i] == 0
                    println("ERROR: The updating atom group is empty at some point.\n ")
                    exit(86)
                end
            end
        end
    end
    println("Atom Group is valid")
    println("Minimum # Atoms: $(minimum(ag_size[ag_used]))")
    println("Maximum # Atoms: $(maximum(ag_size[ag_used]))")
    println("Average # Atoms: $(mean(ag_size[ag_used]))")
    println("────────────────────────────────────────────────────────────────────────────────────")

end

@everywhere function decompose_velocities(v, axis::Vector{Float64})
    # v_ll = (<v|axis>/<axis|axis>)|axis>
    # v_T = v - Proj_axis(v)
    v_ll = ((sum(v'.*axis, dims=1)./sum(axis.*axis)).*axis)'
    v_T = v - v_ll
    return v_T, v_ll
end

function compute_vacf(C::constants)
    # Cv(dt) = <v(t).v(t+dt)>
    #        = <(v_ll(t)+v_T(t)).(v_ll(t+dt)+v_T(t+dt))>
    #        = <v_ll(t).v_ll(t+dt)> + <v_ll(t).v_T(t+dt)> + <v_T(t).v_ll(t+dt)> + <v_T(t).v_T(t+dt)>
    println("COMPUTING VACF")
    vacf = convert(SharedArray, zeros(C.maxdelt+1, 4)) # Create empty array to hold regular acf values

    pool = CachingPool(workers())
    @showprogress pmap(pool, 0:C.maxdelt) do dt
        denom = 0
        for t_i in 1:(C.nframes-dt)
            t = t_i + dt
            for sag in 1:C.nsubag
                sel_t = C.ag[t, :, sag].*C.ag[t_i, :, sag] # atoms in group at both times

                denom += sum(sel_t)
                vel_ti_T, vel_ti_ll = decompose_velocities(C.vel[t_i, sel_t, :], C.ax[t_i, :, sag]) # get components of initial velocities parallel and perpendicular to axis
                vel_t_T, vel_t_ll = decompose_velocities(C.vel[t, sel_t, :], C.ax[t, :, sag]) # get components of future velocities parallel and perpendicular to axis

                vacf[dt+1, 1] += sum(vel_ti_ll.*vel_t_ll) # <v_par(t)|v_par(t+dt)>
                vacf[dt+1, 2] += sum(vel_ti_ll.*vel_t_T) # <v_par(t)|v_perp(t+dt)>
                vacf[dt+1, 3] += sum(vel_ti_T.*vel_t_ll) # <v_perp(t)|v_par(t+dt)>
                vacf[dt+1, 4] += sum(vel_ti_T.*vel_t_T) # <v_perp(t)|v_perp(t+dt)>
            end
        end
        vacf[dt+1, :] = vacf[dt+1, :] ./ denom
    end

    total_vacf = sum(vacf, dims=2)[:,1] # The [:,1] will make it a vector
    println("────────────────────────────────────────────────────────────────────────────────────")

    # Save vacfs
    times = collect(0:C.maxdelt).*C.timestep
    npzwrite(C.output*"vacfs.npz", times=times, total=total_vacf,
             ll_ll=vacf[:,1], ll_T=vacf[:,2], T_ll=vacf[:,3], T_T=vacf[:,4])
    println("VACFs saved to: "*C.output*"vacfs.npz\n")
    println("This NPZ file contains 6 numpy arrays: ")
    println("'times' : a 1D numpy array with the times")
    println("'total' : the total VACF (the sum of the following 4 VACFs)")
    println("'ll_ll' : the VACF parallel to the defined axis(es)")
    println(" 'll_T' : the VACF coupling the parallel velocity at time t_i to the perpendicular "*
            "velocity at time t_i + dt")
    println(" 'T_ll' : the VACF coupling the perpendicular velocity at time t_i to the parallel "*
            "velocity at time t_i + dt")
    println("  'T_T' : the VACF perpendicular to the defined axis(es)\n")
    println("PLEASE NOTE: The coupling VACFs, ll_T and T_ll, will be zero if your axes are unchanging "*
            "and thus can be ignored. If your axes do change, these represent the contributions to the "*
            "total VACF from the rotation of the axes. What you do with these contributions depends on "*
            "your use case. For example, if you are calculating the VACFs of water within a water channel "*
            "that can rotate freely, ignoring these values would tell you about the diffusion of the water "*
            "relative to the channel.")
end

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
function main(parsed_args)
    C, skipcheck = init(parsed_args)

    if skipcheck == false
        evaluate_atomgroup(C)
    end

    compute_vacf(C)
end

main(parsed_args)
