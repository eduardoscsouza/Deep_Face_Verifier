imdb_in_filename = "imdb_crop/imdb.mat"
wiki_in_filename = "wiki_crop/wiki.mat"
imdb_out_filename = "imdb_crop/imdb_idxs.txt"
wiki_out_filename = "wiki_crop/wiki_idxs.txt"

imdb_out_file = fopen(imdb_out_filename, "w");
wiki_out_file = fopen(wiki_out_filename, "w");

load(imdb_in_filename);
for i=1:length(imdb.dob)
    fprintf(imdb_out_file, "%s\t%s\n", imdb.full_path{i}, strrep(strrep(imdb.name{i}, ' ', "_"), "'", ""))
end

load(wiki_in_filename);
for i=1:length(wiki.dob)
    fprintf(wiki_out_file, "%s\t%s\n", wiki.full_path{i}, strrep(strrep(wiki.name{i}, ' ', "_"), "'", ""))
end
