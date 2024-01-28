# -*- coding: utf-8 -*-


from turtle import color
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import PACKAGE_MP_3DLinear as smooth_3d
import PACKAGE_MP_Linear as smooth
import myInput

def plot_energy_figure(timestep, energy_figure, figure_path=None):

    imgs = []
    fig, ax = plt.subplots()

    cv0 = np.squeeze(energy_figure[timestep])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=0,vmax=6,cmap='Accent')
    cb = fig.colorbar(im)
    tx = ax.set_title(f'time step = {timestep}')
    if figure_path != None:
        plt.savefig('energy_{timestep}step')
    plt.show()

def plot_energy_video(timestep, energy_figure, figure_path, delta = 0):

    imgs = []
    fig, ax = plt.subplots()

    cv0 = energy_figure[1,:,:,0] #np.squeeze(energy_figure[0])
    cv0 = np.rot90(cv0,1)
    if delta == 0: colormap_map = np.max(cv0)
    else: colormap_map = (1 + delta) * 8
    im = ax.imshow(cv0,vmin=np.min(cv0),vmax=colormap_map,cmap='Accent',interpolation='none')
    cb = fig.colorbar(im)
    tx = ax.set_title(f'time step = {timestep[0]}')
    # plt.show()

    def animate(i):
        arr=energy_figure[i,:,:,0] #np.squeeze(energy_figure[i])
        arr=np.rot90(arr,1)
        im.set_data(arr)
        tx.set_text(f'time step = {timestep[i]}')

    ani = animation.FuncAnimation(fig, animate, frames=len(timestep))
    FFMpegWriter = animation.writers['ffmpeg']
    writer = animation.FFMpegWriter(fps=math.floor(len(timestep)/5), bitrate=10000)
    ani.save(figure_path+".mp4",writer=writer)

def plot_structure_video(timestep, structure_figure, figure_path):

    imgs = []
    fig, ax = plt.subplots()

    cv0 = structure_figure[1,:,:,0] #np.squeeze(structure_figure[0])
    cv0 = np.rot90(cv0,1)
    im = ax.imshow(cv0,vmin=np.min(cv0),vmax=np.max(cv0),cmap='rainbow',interpolation='none') #jet rainbow plasma
    cb = fig.colorbar(im)
    tx = ax.set_title(f'time step = {timestep[0]}')
    # plt.show()

    def animate(i):
        arr=structure_figure[i,:,:,0] #np.squeeze(structure_figure[i])
        arr=np.rot90(arr,1)
        im.set_data(arr)
        tx.set_text(f'time step = {timestep[i]}')

    ani = animation.FuncAnimation(fig, animate, frames=len(timestep))
    FFMpegWriter = animation.writers['ffmpeg']
    writer = animation.FFMpegWriter(fps=math.floor(len(timestep)/5), bitrate=10000)
    ani.save(figure_path+".mp4",writer=writer)

def dump2img(dump_path, num_steps=None, extract_data='type', extract_step=None):
    # Create grain structure figure from dump file with site ID.
    # if extarct_step is not None, function will extarct one step figrue (extract_step),
    # only work for dump type 1 currently

    # dump file (type 0) or dump.* files (type 1)
    if os.path.exists(dump_path+".dump"):
        dump_type = 0
        dump_file_name_0 = dump_path+".dump"
    elif os.path.exists(dump_path+f".dump.{0 if extract_step == None else int(extract_step)}"):
        dump_type = 1
        dump_file_name_0 = dump_path+f".dump.{0 if extract_step == None else int(extract_step)}"
    else: print("There is no correct dump file for "+dump_path)

    with open(dump_file_name_0) as file:
        box_size = np.zeros(3)
        for i, line in enumerate(file):
            if i==3: num_sites = int(line)
            if i==5: box_size[0] = np.array(line.split(), dtype=float)[-1]
            if i==6: box_size[1] = np.array(line.split(), dtype=float)[-1]
            if i==7: box_size[2] = np.array(line.split(), dtype=float)[-1]
            if i==8: name_vars = line.split()[2:]
            if i>8: break
    box_size = np.ceil(box_size).astype(int) #reformat box_size
    entry_length = num_sites+9 #there are 9 header lines in each entry

    # total lines for dump
    if num_steps!=None: total_lines = num_steps*entry_length
    else: total_lines=None

    time_steps=[]
    grain_structure_figure=[]
    if dump_type == 0:
        with open(dump_file_name_0) as file:
            for i, line in tqdm(enumerate(file), "DECODING (%s.dump)"%dump_path[-20:], total=total_lines):
                [entry_num, line_num] = np.divmod(i,entry_length) #what entry number and entry line number does this line number indicate
                if line_num==0: entry = np.zeros(box_size) #set the energy figure matrix
                if line_num==1: time_steps.append(int(float(line.split()[-1]))) #log the time step
                atom_num = line_num-9 #track which atom line we're on
                if atom_num>=0 and atom_num<num_sites:
                    line_split = np.array(line.split(), dtype=float)
                    site_x = int(line_split[name_vars.index('x')])
                    site_y = int(line_split[name_vars.index('y')])
                    site_z = int(line_split[name_vars.index('z')])
                    entry[site_x,site_y,site_z] = line_split[name_vars.index(extract_data)] #record valid atom lines
                if line_num==entry_length-1:
                    grain_structure_figure.append(entry)
    elif dump_type == 1:
        if extract_step == None:
            dump_item = 0
            dump_file_name_item = dump_path+".dump."+str(int(dump_item))
            while os.path.exists(dump_file_name_item):
                with open(dump_file_name_item) as file:
                    for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%dump_file_name_item[-20:], total=entry_length):
                        if i==0: entry = np.zeros(box_size) #set the energy figure matrix
                        if i==1: time_steps.append(int(float(line.split()[-1]))) #log the time step
                        atom_num = i-9 #track which atom line we're on
                        if atom_num>=0 and atom_num<num_sites:
                            line_split = np.array(line.split(), dtype=float)
                            site_x = int(line_split[name_vars.index('x')])
                            site_y = int(line_split[name_vars.index('y')])
                            site_z = int(line_split[name_vars.index('z')])
                            entry[site_x,site_y,site_z] = line_split[name_vars.index(extract_data)] #record valid atom lines
                        if i==entry_length-1: grain_structure_figure.append(entry)
                # jump to next dump.*
                dump_item += 1
                dump_file_name_item = dump_path+".dump."+str(int(dump_item))
        else:
            # extarct only specific steps
            dump_item = int(extract_step)
            dump_file_name_item = dump_path+".dump."+str(int(dump_item))
            with open(dump_file_name_item) as file:
                for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%dump_file_name_item[-20:], total=entry_length):
                    if i==0: entry = np.zeros(box_size) #set the energy figure matrix
                    if i==1: time_steps.append(int(float(line.split()[-1]))) #log the time step
                    atom_num = i-9 #track which atom line we're on
                    if atom_num>=0 and atom_num<num_sites:
                        line_split = np.array(line.split(), dtype=float)
                        site_x = int(line_split[name_vars.index('x')])
                        site_y = int(line_split[name_vars.index('y')])
                        site_z = int(line_split[name_vars.index('z')])
                        entry[site_x,site_y,site_z] = line_split[name_vars.index(extract_data)] #record valid atom lines
                    if i==entry_length-1: grain_structure_figure.append(entry)
    else: print("wrong dump file input!")
    grain_structure_figure = np.array(grain_structure_figure)
    time_steps = np.array(time_steps)

    return time_steps, grain_structure_figure

def get_normal_vector(grain_structure_figure_one):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one
    R = np.zeros((nx,ny,2))
    smooth_class = smooth.linear_class(nx,ny,ng,cores,loop_times,P0,R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together, sites

def get_normal_vector_3d(grain_structure_figure_one):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    nz = grain_structure_figure_one.shape[2]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one[:,:,:,np.newaxis]
    R = np.zeros((nx,ny,nz,3))
    smooth_class = smooth_3d.linear3d_class(nx,ny,nz,ng,cores,loop_times,P0,R,'p')

    smooth_class.linear3d_main("inclination")
    P = smooth_class.get_P()
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print(f"Total num of GB sites: {len(sites_together)}")

    return P, sites_together, sites

def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j] = sitei
        dx,dy = myInput.get_grad(P,i,j)
        degree.append(math.atan2(-dy, dx) + math.pi)
    for i in range(len(degree)):
        freqArray[int((degree[i]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    # bias situation
    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)
    # Plot
    plt.plot(xCor/180*math.pi, freqArray, linewidth=2, label=para_name)

    return 0

def get_normal_vector_slope_3d(P, sites, step, para_name, angle_index=0, bias=None):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    xCor = np.linspace((xLim[0]+binValue/2),(xLim[1]-binValue/2),binNum)

    freqArray = np.zeros(binNum)
    degree = []
    for sitei in sites:
        [i,j,k] = sitei
        dx,dy,dz = myInput.get_grad3d(P,i,j,k)
        if angle_index == 0:
            dx_fake = dx
            dy_fake = dy
        elif angle_index == 1:
            dx_fake = dx
            dy_fake = dz
        elif angle_index == 2:
            dx_fake = dy
            dy_fake = dz

        # Normalize
        if math.sqrt(dy_fake**2+dx_fake**2) < 1e-5: continue
        dy_fake_norm = dy_fake / math.sqrt(dy_fake**2+dx_fake**2)
        dx_fake_norm = dx_fake / math.sqrt(dy_fake**2+dx_fake**2)

        degree.append(math.atan2(-dy_fake_norm, dx_fake_norm) + math.pi)
    for n in range(len(degree)):
        freqArray[int((degree[n]/math.pi*180-xLim[0])/binValue)] += 1
    freqArray = freqArray/sum(freqArray*binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray/sum(freqArray*binValue)

    # Plot
    plt.plot(np.append(xCor,xCor[0])/180*math.pi, np.append(freqArray,freqArray[0]),linewidth=2,label=para_name)

    return freqArray

def simple_magnitude(freqArray):
    # Get the simple anisotropic magnitude from inclination distribution

    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0])+abs(xLim[1]))/binValue)
    # prefect circle
    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle/sum(freqArray_circle*binValue)
    # max/average(difference between currect distribuition and perfect distribution) over average of perfect distribution
    magnitude_max = np.max(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle))/np.average(freqArray_circle)
    # standard
    magnitude_stan = np.sqrt(np.sum((abs(freqArray - freqArray_circle)/np.average(freqArray_circle) - magnitude_ave)**2)/binNum)

    return magnitude_ave, magnitude_stan


def calculate_expected_step(input_npy_data, expected_grain_num=200):
    # calculate the timestep for expected_grain_num in npy dataset

    num_input = len(input_npy_data)
    special_step_distribution = np.zeros(num_input)
    microstructure_list = []

    for input_i in range(num_input):
        npy_data = np.load(input_npy_data[input_i])
        step_num = npy_data.shape[0]
        grain_num_list = np.zeros(step_num)
        for i in tqdm(range(step_num)):
            grain_num_list[i] = len(set(npy_data[i,:].flatten()))
        special_step_distribution[input_i] = int(np.argmin(abs(grain_num_list - expected_grain_num)))
        microstructure_list.append(npy_data[int(special_step_distribution[input_i]),:])
    print("> Step calculation done")

    return special_step_distribution, microstructure_list


def get_poly_center(micro_matrix, step):
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step,:]))
    center_list = np.zeros((num_grains,2))
    sites_num_list = np.zeros(num_grains)
    ave_radius_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    table = micro_matrix[step,:,:,0]
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i+1)

        if (sites_num_list[i] < 500) or \
           (np.max(coord_refer_i[table == i+1]) - np.min(coord_refer_i[table == i+1]) == micro_matrix.shape[1]) or \
           (np.max(coord_refer_j[table == i+1]) - np.min(coord_refer_j[table == i+1]) == micro_matrix.shape[2]): # grains on bc are ignored
          center_list[i, 0] = 0
          center_list[i, 1] = 0
          sites_num_list[i] == 0
        else:
          center_list[i, 0] = np.sum(coord_refer_i[table == i+1]) / sites_num_list[i]
          center_list[i, 1] = np.sum(coord_refer_j[table == i+1]) / sites_num_list[i]
    ave_radius_list = np.sqrt(sites_num_list / np.pi)

    return center_list, ave_radius_list

def get_poly_statistical_radius(micro_matrix, sites_list, step):
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step,:]))

    max_radius_offset_list = np.zeros(num_grains)
    for n in range(num_grains):
        center = center_list[n]
        ave_radius = ave_radius_list[n]
        sites = sites_list[n]

        if ave_radius != 0:
          for sitei in sites:
              [i,j] = sitei
              current_radius = np.sqrt((i - center[0])**2 + (j - center[1])**2)
              radius_offset = abs(current_radius - ave_radius)
              if radius_offset > max_radius_offset_list[n]: max_radius_offset_list[n] = radius_offset

          max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    area_list = np.pi*ave_radius_list*ave_radius_list
    if np.sum(area_list) == 0: max_radius_offset = 0
    else: max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset

def get_poly_statistical_ar(micro_matrix, step):
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step,:]))
    sites_num_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i,j] = i
            coord_refer_j[i,j] = j

    aspect_ratio_i = np.zeros((num_grains,2))
    aspect_ratio_j = np.zeros((num_grains,2))
    aspect_ratio = np.zeros(num_grains)
    table = micro_matrix[step,:,:,0]

    aspect_ratio_i_list = [[] for _ in range(int(num_grains))]
    aspect_ratio_j_list = [[] for _ in range(int(num_grains))]
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(table[i][j]-1)
            sites_num_list[grain_id] +=1
            aspect_ratio_i_list[grain_id].append(coord_refer_i[i][j])
            aspect_ratio_j_list[grain_id].append(coord_refer_j[i][j])

    for i in range(num_grains):
        aspect_ratio_i[i, 0] = len(list(set(aspect_ratio_i_list[i])))
        aspect_ratio_j[i, 1] = len(list(set(aspect_ratio_j_list[i])))
        if aspect_ratio_j[i, 1] == 0: aspect_ratio[i] = 0
        else: aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]

    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio

def init2EAarray(init_file_path, grain_num):
    euler_angle_array = np.ones((grain_num, 3))*-2
    with open(init_file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i > 2: euler_angle_array[int(line.split()[1])-1] = np.array([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
    # Check if missing grains
    for i in range(grain_num):
        if euler_angle_array[i,0] == -2:
            print("Missing grains here!")
            euler_angle_array[i] = np.array([0,0,0])
    return euler_angle_array

def output_init_from_dump(dump_file_path, euler_angle_array, init_file_path_output):
    # output the init file with euler_angle_array and one dump file
    # Read necessary information from dump file
    with open(dump_file_path) as file:
        box_size = np.zeros(3)
        for i, line in enumerate(file):
            if i==3: num_sites = int(line)
            if i==5: box_size[0] = np.array(line.split(), dtype=float)[-1]
            if i==6: box_size[1] = np.array(line.split(), dtype=float)[-1]
            if i==7: box_size[2] = np.array(line.split(), dtype=float)[-1]
            if i==8: name_vars = line.split()[2:]
            if i>8: break
    box_size = np.ceil(box_size).astype(int) #reformat box_size
    entry_length = num_sites+9 #there are 9 header lines in each entry


    # write the IC files and read the dump data
    # create IC file
    IC_nei = []
    IC_nei.append("# This line is ignored\n")
    IC_nei.append("Values\n")
    IC_nei.append("\n")
    with open(init_file_path_output, 'w') as output_file:
        output_file.writelines( IC_nei )
    IC_nei = []
    # read and write
    with open(init_file_path_output, 'a') as output_file:
        with open(dump_file_path) as file:
            for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%dump_file_path[-20:], total=entry_length):
                if i==1: time_step = int(float(line.split()[-1])) #log the time step
                atom_num = i-9 #track which atom line we're on
                if atom_num>=0 and atom_num<num_sites:
                    line_split = np.array(line.split(), dtype=float)
                    grain_id = int(line_split[1])-1
                    output_file.write(f"{int(line_split[0])} {int(line_split[1])} {euler_angle_array[grain_id, 0]} {euler_angle_array[grain_id, 0]} {euler_angle_array[grain_id, 0]}\n")

    return box_size, entry_length

def output_init_neighbor_from_init(interval, box_size, init_file_path_input, init_file_path_output):
    # Output the init_nighbor5 with init file

    nei_num = (2*interval+3)**3-1
    size_x,size_y,size_z = box_size
    img = np.zeros((size_y,size_x,size_z)) #Figure of all sites with GrainID

    print(f"> img matrix start.")
    for k in tqdm(range(size_z)): # z-axis
        for i in range(size_y): # y-axis
            for j in range(size_x): # x-axis
                img[i,j,k] = int(k*size_x*size_y + i*size_x + j)
    print(f"> img matrix end")

    IC_nei = []
    IC_nei.append("# This line is ignored\n")
    IC_nei.append("3 dimension\n")
    IC_nei.append(f"{nei_num} max neighbors\n")
    IC_nei.append(f"{size_x*size_y*size_z} sites\n")
    IC_nei.append(f"0 {size_x} xlo xhi\n")
    IC_nei.append(f"0 {size_y} ylo yhi\n")
    IC_nei.append(f"0 {size_z} zlo zhi\n")
    IC_nei.append("\n")
    IC_nei.append("Sites\n")
    IC_nei.append("\n")
    with open(init_file_path_output, 'w') as file:
        file.writelines( IC_nei )
    IC_nei = []

    print("> Sites start writing")
    with open(init_file_path_output, 'a') as file:
        for k in tqdm(range(size_z)): # z-axis
            for i in range(size_y): # y-axis
                for j in range(size_x): # x-axis
                    file.write(f"{int(img[i,j,k] + 1)} {float(j)} {float(i)} {float(k)}\n")
    print("> Sites end writing")

    IC_nei.append("\n")
    IC_nei.append("Neighbors\n")
    IC_nei.append("\n")
    with open(init_file_path_output, 'a') as file:
        file.writelines( IC_nei )
    IC_nei = []

    print("> Neighbors start writing")
    max_length_neighbors = 0
    with open(init_file_path_output, 'a') as file:
        for k in tqdm(range(size_z)): # z-axis
            for i in range(size_y): # y-axis
                for j in range(size_x): # x-axis
                    tmp_nei = f"{int(img[i,j,k] + 1)} "
                    offsets = np.array(np.meshgrid(
                    np.arange(-(interval + 1), interval + 2),
                    np.arange(-(interval + 1), interval + 2),
                    np.arange(-(interval + 1), interval + 2),
                    )).T.reshape(-1, 3)
                    # Filter out the [0, 0, 0] offset since we want to skip it
                    offsets = offsets[np.any(offsets != 0, axis=1)]
                    # Compute the indices with wrapping around boundaries (using np.mod)
                    indices = (np.array([i, j, k]) + offsets) % np.array([size_y, size_x, size_z])
                    # Extract the values from 'img' using advanced indexing
                    neighbour_values = img[indices[:, 0], indices[:, 1], indices[:, 2]].astype('int')
                    # Convert values to 1-based indexing and concatenate into a string
                    tmp_nei += ' '.join(map(str, neighbour_values + 1))
                    # tmp_nei = f"{int(img[i,j,k] + 1)}"
                    # for p in range(-(interval+1),interval+2):
                    #     for m in range(-(interval+1),interval+2):
                    #         for n in range(-(interval+1),interval+2):
                    #             if m==0 and n==0 and p==0: continue
                    #             tmp_i = (i+m)%size_y
                    #             tmp_j = (j+n)%size_x
                    #             tmp_k = (k+p)%size_z
                    #             tmp_nei += f" {int(img[tmp_i, tmp_j, tmp_k]+1)}"

                    # IC_nei.append(tmp_nei+"\n")
                    if len(tmp_nei) > max_length_neighbors: max_length_neighbors = len(tmp_nei)
                    file.write(tmp_nei+"\n")
        file.write("\n")
    print(f"The max length of neighbor data line is {max_length_neighbors}")
    print("> Neighbors end writing")

    print("> Values start writing")
    with open(init_file_path_input, 'r') as f_read:
        tmp_values = f_read.readlines()
    print("> Values read done")
    with open(init_file_path_output, 'a') as file:
        file.writelines(tmp_values[1:])
    print("> Values end writing")
    return True





if __name__ == '__main__':
    path  = "/blue/michael.tonks/lin.yang/SPPARKS-VirtualIncEnergy/3d_poly/"
    file_name = ["p_ori_ave_aveE_100_20k_multiCore64_delta0.8_m2_J1_refer_1_0_0_seed56689_kt066"]
    figure_path = file_name
    num_grain = 20000
    output_name = ["ave_02_distribution"]

    for i in range(len(file_name)):
        if not os.path.exists("results/"+figure_path[i]+".npy"):
            timestep, grain_structure_figure = dump2img(path+file_name[i], 81)
            np.save("results/"+figure_path[i],grain_structure_figure)
        else:
            grain_structure_figure = np.load(path + "results/"+figure_path[i]+".npy")
            timestep = 1.0 * np.array(range(len(grain_structure_figure)))

        if not os.path.exists("results/"+figure_path[i]): os.mkdir("results/"+figure_path[i])
        slope_list = np.zeros(len(timestep))
        for step in range(len(timestep)):
            newplace = np.rot90(grain_structure_figure[step,:,:,:], 1, (0,1))
            P, sites = get_normal_vector_3d(newplace)
            if len(sites) == 0: continue
            slope_list[step] = plot_normal_vector_distribution_3d(P, sites, step, figure_path[i]+"/"+figure_path[i])

        os.system(f'ffmpeg -framerate 30 -i results/{figure_path[i]}/{figure_path[i]}_step.%04d.png \
                    -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p \
                    results/{figure_path[i]}/{output_name[i]}.mp4')
        plt.clf()
        fig = plt.subplots()
        # plt.bar(xCor,freqArray,width=binValue*0.7)
        plt.plot(timestep, slope_list*1e4,'-o',linewidth=2,label='slope of normal distribution')
        plt.xlabel("Timestep")
        plt.ylabel("Slope/1e-4")
        plt.ylim([-20, 20])
        plt.legend()
        plt.savefig(f'results/normal_distribution_slope/{figure_path[i]}_slope.png',dpi=400,bbox_inches='tight')
