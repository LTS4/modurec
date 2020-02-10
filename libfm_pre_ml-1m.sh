./libfm-1.40.windows/convert \
--ifile data/ml-1m/libfm/rel.item.libfm \
--ofilex data/ml-1m/libfm/rel.item.x \
--ofiley temp

./libfm-1.40.windows/convert \
--ifile data/ml-1m/libfm/rel.user.libfm \
--ofilex data/ml-1m/libfm/rel.user.x \
--ofiley temp

./libfm-1.40.windows/convert \
--ifile data/ml-1m/libfm/rel.time.libfm \
--ofilex data/ml-1m/libfm/rel.time.x \
--ofiley temp

./libfm-1.40.windows/transpose \
--ifile data/ml-1m/libfm/rel.item.x \
--ofile data/ml-1m/libfm/rel.item.xt

./libfm-1.40.windows/transpose \
--ifile data/ml-1m/libfm/rel.user.x \
--ofile data/ml-1m/libfm/rel.user.xt

./libfm-1.40.windows/transpose \
--ifile data/ml-1m/libfm/rel.time.x \
--ofile data/ml-1m/libfm/rel.time.xt