import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
import pickle
import numpy as np

#np.random.seed(10)
torch.manual_seed(10)

# ================== Normalization ==================
def normalize_seq(frames, fps, gap, option, widths, heights):
    coords = np.array([f for f in frames], dtype=np.float32)
    T = coords.shape[0]
    ref = coords[0]
    if option == 5:
        x, y = coords.T
    else:
        x, y, w, h = coords.T

    if option == 1:
        norm = np.stack([(x[:] - ref[0]) / w[:],
                         (y[:] - ref[1]) / h[:],
                         (w[:] - ref[2]) / w[:],
                         (h[:] - ref[3]) / h[:]], axis=1)
        delta = np.empty([2, 2])
    elif option == 2:
        ref_time = 1 / int(fps[0])
        num_sort = np.arange(2, T + 1).reshape(-1, 1)
        r_time = num_sort / fps
        delta = (r_time - ref_time).squeeze()
        norm = np.stack([(x[1:] - ref[0]) / (delta * w[1:]),
                         (y[1:] - ref[1]) / (delta * h[1:]),
                         (w[1:] - ref[2]) / (delta * w[1:]),
                         (h[1:] - ref[3]) / (delta * h[1:])], axis=1)
        norm = np.concatenate([[[0, 0, 0, 0]], norm], axis=0)
    elif option == 3:
        norm = np.stack([(x[:] - ref[0]) / h[:],
                         (y[:] - ref[1]) / h[:],
                         (w[:] - ref[2]) / h[:],
                         (h[:] - ref[3]) / h[:]], axis=1)
        delta = np.empty([2, 2])
    elif option == 4:
        cumsum = np.cumsum(h)
        time = np.arange(2, h.shape[0] + 1).reshape(-1, 1)
        moving_avg_h = cumsum[1:] / time.squeeze()

        cumsum = np.cumsum(w)
        time = np.arange(2, w.shape[0] + 1).reshape(-1, 1)
        moving_avg_w = cumsum[1:] / time.squeeze()

        norm = np.stack([(x[1:] - ref[0]) / moving_avg_w,
                         (y[1:] - ref[1]) / moving_avg_h,
                         (w[1:] - ref[2]) / moving_avg_w,
                         (h[1:] - ref[3]) / moving_avg_h], axis=1)
        norm = np.concatenate([[[0, 0, 0, 0]], norm], axis=0)
        delta_w = np.concatenate((w[:1], moving_avg_w), axis=0)
        delta_h = np.concatenate((h[:1], moving_avg_h), axis=0)
        delta = np.stack([delta_w, delta_h], axis=1)
    elif option == 5:
        norm = np.stack((x[:] / widths.squeeze(), y[:] / heights.squeeze()), axis=1)
        delta = np.empty([2, 2])
    elif option == 6:
        norm = np.stack([x[:] / widths, y[:] / heights,
                         w[:] / widths, h[:] / heights], axis=1)
        delta = np.empty([2, 2])
    elif option == 7:
        ref_time = 1 / int(fps[0])
        num_sort = np.arange(2, T + 1).reshape(-1, 1)
        r_time = num_sort / fps
        delta = (r_time - ref_time).squeeze()
        norm = np.stack([(x[1:] - ref[0]) / delta,
                         (y[1:] - ref[1]) / delta,
                         (w[1:] - ref[2]) / delta,
                         (h[1:] - ref[3]) / delta], axis=1)
        norm = np.concatenate([[[0, 0, 0, 0]], norm], axis=0)
    else:
        raise ValueError("Invalid option")

    return norm, ref, coords[-gap:], coords[:-gap], delta


# ================== Dataset ==================
class TrajDataset(Dataset):
    def __init__(self, pickle_file, args, option=1, min_len=1, max_gap=1, max_overlap=0.4, augment=True):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        self.option = option
        gap = args.target_len
        self.index = []

        for rec_id, rec in enumerate(self.data):
            frames = rec['frames']  # list of (frame_num, [x,y,w,h])
            fps = rec['fps']
            T = len(frames)
            if T < args.min_len + gap + 1:
                continue

            frame_nums = [f[0] for f in frames]

            def has_valid_coords(frame):
                coords = frame[1]
                return all(np.isfinite(coords)) and any(c > 0 for c in coords)

            max_seq_len = min(args.max_len, T - gap - 1)
            for seq_len in range(args.min_len, max_seq_len + 1):
                stride = max(1, int(seq_len * (1 - max_overlap)))
                for start_idx in range(0, T - seq_len - gap, stride):
                    end_idx = start_idx + seq_len
                    target_idx = end_idx + gap
                    if target_idx >= T:
                        continue

                    # consecutive check (no missing frames)
                    seq_frame_nums = frame_nums[start_idx:target_idx + 1]
                    #print(seq_frame_nums)
                    if not np.all(np.diff(seq_frame_nums) == 1):
                        continue

                    if not (has_valid_coords(frames[start_idx]) and
                            has_valid_coords(frames[end_idx - 1]) and
                            has_valid_coords(frames[target_idx])):
                        continue
                    #print(seq_frame_nums)
                    self.index.append((rec_id, start_idx, seq_len, gap))

                    if augment:
                        if 'gen' in pickle_file:
                            self.index.append((rec_id, start_idx, seq_len, gap, "scale"))
                            self.index.append((rec_id, start_idx, seq_len, gap, "translate"))
                        else:
                        #print(augment)
                        #self.index.append((rec_id, start_idx, seq_len, gap))
                            self.index.append((rec_id, start_idx, seq_len, gap, "scale"))
                            self.index.append((rec_id, start_idx, seq_len, gap, "translate"))
                        #self.index.append((rec_id, start_idx, seq_len, gap, "scale"))
                        #self.index.append((rec_id, start_idx, seq_len, gap, "translate"))
                        #self.index.append((rec_id, start_idx, seq_len, gap, "scale"))
                        #self.index.append((rec_id, start_idx, seq_len, gap, "translate"))

        print(f"{len(self.index)} samples from {pickle_file}", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if len(self.index[idx]) == 4:
            rec_id, start_idx, seq_len, gap = self.index[idx]
            aug_type = None
        else:
            rec_id, start_idx, seq_len, gap, aug_type = self.index[idx]

        #rec_id, start_idx, seq_len, gap = self.index[idx]
        rec = self.data[rec_id]
        frames = rec['frames']
        fps = rec['fps']
        width = np.array([rec['width']])
        height = np.array([rec['height']])
        sequence = rec['sequence']
        old_track_id = rec['old_track_id']

        end_idx = start_idx + seq_len
        target_idx_end = end_idx + gap

        # inputs and targets (guaranteed consecutive)
        input_coords = [f[1] for f in frames[start_idx:end_idx]]
        target_frames = [f[1] for f in frames[end_idx:target_idx_end]]

        all_coords = np.array(input_coords + target_frames, dtype=np.float32)  # (T,4)
        if self.option == 5:
           xy = all_coords[:, :2]
        else:
           xy, wh = all_coords[:, :2], all_coords[:, 2:]

        # =========================
        # Apply augmentation
        # =========================
        if aug_type == "rotate":
            theta = np.random.uniform(-15, 15) * np.pi / 180  # ±15°
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
            xy = xy @ R.T
        elif aug_type == "scale":
            s = np.random.uniform(0.9, 1.1)  # zoom ±10%
            #xy *= s
            #if self.option != 5:
            wh *= s
        elif aug_type == "translate":
            shift = np.random.uniform(-0.05, 0.05, size=(1, 2))  # shift ±5% (normalized units)
            xy += shift
        elif aug_type == "speed":
            displacements = np.diff(traj, axis=0)

            scale = 1 + np.random.normal(0, strength, size=(len(displacements),1))
            displacements = displacements * scale

            out = traj.copy()
            out[1:] = out[0] + np.cumsum(displacements, axis=0)
        if self.option !=5:
           all_coords = np.concatenate([xy, wh], axis=1)
        else:
           all_coords = xy
        input_coords = all_coords[:len(input_coords)]
        target_frames = all_coords[len(input_coords):]


        inp, ref, last_abs, inp_abs, delta = normalize_seq(
            input_coords.tolist() + target_frames.tolist(), fps, gap, self.option, width, height
        )

        delta_inp = delta[:-gap]
        delta_tgt = delta[-gap:]

        tgt = inp[-gap:]
        frame_nums = [f[0] for f in frames[start_idx:target_idx_end]]

        # time to target
        #print(fps)
        if isinstance(fps, float):
            m = gap / int(fps)
        else:
            m = gap / int(fps[0])

        return (
            torch.from_numpy(inp[:-gap]).float(),
            torch.from_numpy(tgt).float(),
            torch.from_numpy(ref).float(),
            torch.from_numpy(last_abs).float(),
            torch.from_numpy(inp_abs).float(),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(start_idx, dtype=torch.int32),
            torch.tensor(end_idx, dtype=torch.int32),
            torch.tensor(target_idx_end, dtype=torch.int32),
            torch.tensor(delta_inp, dtype=torch.float32),
            torch.tensor(delta_tgt, dtype=torch.float32),
            np.array(sequence, dtype=str),
            np.array(old_track_id, dtype=str),
            torch.tensor(frame_nums, dtype=torch.int32),
            np.array(width, dtype=int),
            np.array(height, dtype=int)
        )


# ================== Collate ==================
def collate_fn(batch):
    inp, tgt, ref, last, inp_abs, m, start_idx, end_idx, target_idx, delta_inp, delta_tgt, sequence, old_track_id, frame_num, width, height = zip(*batch)

    lengths = [s.shape[0] for s in inp]
    inp_p = rnn_utils.pad_sequence(inp, batch_first=True, padding_value=0).float()
    inp_abs = rnn_utils.pad_sequence(inp_abs, batch_first=True, padding_value=0).float()
    tgt_p = torch.stack(tgt)
    ref = torch.stack(ref)
    last = torch.stack(last)
    m = torch.stack(m)
    start_idx = torch.stack(start_idx)
    end_idx = torch.stack(end_idx)
    target_idx = torch.stack(target_idx)
    delta_inp = rnn_utils.pad_sequence(delta_inp, batch_first=True, padding_value=0).float()
    delta_tgt = torch.stack(delta_tgt)
    frame_num = rnn_utils.pad_sequence(frame_num, batch_first=True, padding_value=0).int()
    sequence = np.stack(sequence)
    old_track_id = np.stack(old_track_id)
    width = np.stack(width)
    height = np.stack(height)

    return inp_p, tgt_p, lengths, ref, last, inp_abs, m, start_idx, end_idx, target_idx, delta_inp, delta_tgt, sequence, old_track_id, frame_num, width, height

