import numpy as np

rng = np.random.default_rng()

def lazy_2pt(data):
    Npoints = len(data)
    big_dist = np.zeros((Npoints, Npoints))
    for i, d in enumerate(data):
        big_dist[i,:] = np.linalg.norm(data - d, axis=1)
    return big_dist

def less_lazy_2pt(data):
    flat_dists = []
    for i,d in enumerate(data):
        flat_dists.append(np.linalg.norm(data[i+1:] - d, axis=1))
    flat_dists = np.concatenate(flat_dists)
    return flat_dists

def twosamp_2pt(data, otherset):
    flat_dists = []
    for i,d in enumerate(data):
        flat_dists.append(np.linalg.norm(otherset - d, axis=1))
    flat_dists = np.concatenate(flat_dists)
    return flat_dists

def binned_twosamp_2pt(data, otherset, bins):
    xi = np.zeros(len(bins)-1)
    dlen = len(data)
    for i,d in enumerate(data):
        if (i%1000)==0:
            print(f"On run {i}/{dlen} ~ {round(i/dlen, 2)}")
        flat_dists = np.linalg.norm(otherset - d, axis=1)
        xi += np.histogram(flat_dists, bins=bins)[0]
    return xi

def mp_binned_twosamp_2pt(data, otherset, bins, ndxs, q):
    xi = np.zeros(len(bins)-1)
    for ndx in ndxs:
        d = data[ndx]
        flat_dists = np.linalg.norm(otherset - d, axis=1)
        xi += np.histogram(flat_dists, bins=bins)[0]
    q.put(xi)
    return xi

def gen_points(Npoints, pdist_xs, pdist_cutoffs, box_size=10):
    new_data = np.zeros((Npoints, 3))
    new_data[0] = np.ones(3) * box_size//2

    start_pt = new_data[0]

    for i in range(Npoints-1):
        r = pdist_xs[:-1][rng.uniform() > pdist_cutoffs][-1]
        u,v = rng.uniform(size=2)
        phi = 2*np.pi*u
        theta = np.arccos(2*v - 1)

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        add_vec = np.array([x,y,z])

        new_pt = (start_pt + add_vec) % 10
        new_data[i+1] = new_pt
        start_pt = new_pt
    return new_data

