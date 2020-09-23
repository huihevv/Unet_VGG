from tensorboard.backend.event_processing import event_accumulator

#加载日志数据
ea = event_accumulator.EventAccumulator("./tmp/log/events.out.tfevents.1545401625.ed1b7e66a8b7")
ea.Reload()
print(ea.scalars.Keys())

# val_loss = ea.scalars.Items('val_loss')
# print(len(val_loss))
# print([(i.step, i.value) for i in val_loss])

loss = ea.scalars.Items('loss')
print(len(loss))
print([(i.step, i.value) for i in loss])

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)

# val_acc = ea.scalars.Items('val_acc')
# ax1.plot([i.step for i in val_acc], [i.value for i in val_acc], label='val_acc')
# ax1.set_xlim(0)

# val_loss = ea.scalars.Items('val_loss')
# ax1.plot([i.step for i in val_loss], [i.value for i in val_loss], label='val_loss')
# ax1.set_xlim(0)

# acc = ea.scalars.Items('acc')
# ax1.plot([i.step for i in acc], [i.value for i in acc], label='acc')
# ax1.set_xlim(0)

loss = ea.scalars.Items('loss')
ax1.plot([i.step for i in loss], [i.value for i in loss], label='loss')
ax1.set_xlim(0)

ax1.set_xlabel("epoch")
ax1.set_ylabel("loss function")

plt.legend(loc='lower right')
plt.show()
