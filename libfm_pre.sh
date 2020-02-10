./libfm-1.40.windows/convert \
--ifile data/ml-100k/libfm/rel.item.libfm \
--ofilex data/ml-100k/libfm/rel.item.x \
--ofiley temp

./libfm-1.40.windows/convert \
--ifile data/ml-100k/libfm/rel.user.libfm \
--ofilex data/ml-100k/libfm/rel.user.x \
--ofiley temp

./libfm-1.40.windows/convert \
--ifile data/ml-100k/libfm/rel.time.libfm \
--ofilex data/ml-100k/libfm/rel.time.x \
--ofiley temp

./libfm-1.40.windows/transpose \
--ifile data/ml-100k/libfm/rel.item.x \
--ofile data/ml-100k/libfm/rel.item.xt

./libfm-1.40.windows/transpose \
--ifile data/ml-100k/libfm/rel.user.x \
--ofile data/ml-100k/libfm/rel.user.xt

./libfm-1.40.windows/transpose \
--ifile data/ml-100k/libfm/rel.time.x \
--ofile data/ml-100k/libfm/rel.time.xt