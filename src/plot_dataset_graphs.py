import matplotlib.pyplot as plt

labels1 = ['Real (FFHQ)', 'Fake (GANs)']
real_count = 10000
fake_counts = [2500, 2500, 2500, 2500]  # PGGANv1, PGGANv2, StyleGAN_FFHQ, StyleGAN_CelebA
fake_labels = ['PGGANv1', 'PGGANv2', 'StyleGAN_FFHQ', 'StyleGAN_CelebA']
fake_colors = ['red', 'darkred', 'crimson', 'firebrick']

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(labels1[0], real_count, color='blue', label='Real (FFHQ)')
bottom = 0
for i in range(len(fake_counts)):
    ax.bar(labels1[1], fake_counts[i], bottom=bottom, color=fake_colors[i], label=fake_labels[i])
    bottom += fake_counts[i]
bottom = 0
for i in range(len(fake_counts)):
    ax.text(1, bottom + fake_counts[i] / 2, fake_labels[i], ha='center', va='center', fontsize=16, color='white', fontweight='bold')
    bottom += fake_counts[i]

ax.set_ylabel('Number of Images', fontsize=16)
ax.set_title('Distribution of Real and Fake Images', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
plt.ylim(0, 16000)
plt.legend(loc='upper left', fontsize=14)
plt.show()

labels2 = ['Validation (15%)', 'Training (70%)', 'Test (15%)']
total_images = 20000
split_percentages = [0.15, 0.7, 0.15]
split_counts = [int(p * total_images) for p in split_percentages]
real_counts = [count // 2 for count in split_counts]
fake_counts = real_counts

split_colors = ['green', 'limegreen']
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(3):
    bottom = 0
    ax.bar(labels2[i], real_counts[i], bottom=bottom, color=split_colors[0], label='Real' if i == 0 else "")
    bottom += real_counts[i]
    ax.bar(labels2[i], fake_counts[i], bottom=bottom, color=split_colors[1], label='Fake' if i == 0 else "")
for i in range(3):
    ax.text(i, split_counts[i] / 2, f"{split_counts[i]}\n(50% real, \n50% fake)",
            ha='center', va='center', fontsize=16, color='white', fontweight='bold')

ax.set_ylabel('Number of Images', fontsize=16)
ax.set_title('Dataset Split: Train, Validation, Test', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
plt.ylim(0, 16000)
plt.legend(loc='upper right', fontsize=14)
plt.show()