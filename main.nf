#!/usr/bin/env nextflow
if (!params.get('raw_reads', "")) {
    def templateString = params.config_template
    def template = file('config_tempalte.yml')
    template.write(templateString)
    exit 0, """Please specify your configuration file. e.g
               > nextflow run zhqu1148980644/hictools -params-file your_config.yml -resume  
               You can find a configuration template in current directory.             
            """
}


/*************************************parameters*********************************/ 
raw_reads = params.raw_reads
Boolean do_stat = params.stat
Boolean do_fastqc = params.fastqc
def indexs = file(params.index + "*", checkIfExists: true)
if (indexs.size() == 0)
    error "Index cant not be found: ${params.index}"
def index_path = indexs[0].getParent()
def index_prefix = params.index.replaceAll(".*/", "")

// map parameters
def assembly = params.map.assembly
def chunksize = params.map.chunksize * 4
Boolean merge_bam = params.map.merge_bam
def genome = params.map.genome

// parse
def parse_args = params.parse.parse_args
def sort_tmpdir = params.parse.sort_tmpdir

// filtering
def frag_path = params.filter.frag_path
def enzyme = params.filter.enzyme
def select_expr = params.filter.filter_expr
Boolean do_restrict = frag_path || enzyme

// binning
def resolutions = params.resolutions
def min_res = resolutions.collect{it as int}.min()

// Channels
Channel.value([file(index_path), index_prefix]).set {index}
Channel.value(file(params.chromsize, checkIfExists: true)).set {chromsize}

if (genome)
    Channel.value(file(genome, checkIfExists: true)).set {fasta}
else
    Channel.empty().set {fasta}

if (frag_path)
    Channel.from(file(frag_path, checkIfExists: true)).set {local_frag}
else
    Channel.from("").set {local_frag}

Channel.from(enzyme).set {enzyme_name}

if (!frag_path && enzyme && !genome)
    error " GBenome file needed for cutting with enzyme: ${enzyme}"

// cpus ratio
def thread(task) {
    def ratio = params.get('ratio', 1.0) as float
    ratio = Math.max(ratio, 1.0)
    if (task.cpus != null)
        return Math.max((task.cpus * ratio) as int, 1)
    else
        return Math.max(ratio as int, 1)
}
/****************************************utils*************************************************/ 

String commonPrefix(s1, s2) {
    def common = StringUtils.getCommonPrefix(s1, s2)
    if ((s1 ==~ /.*_[1,2]\..*/) 
        && (s2 ==~ /.*_[1,2]\..*/)
        && (common[-1] == '_'))
        return common[0..-2]
    else {
        def ss1 = s1 =~ /(.*?)\..*/
        def ss2 = s2 =~ /(.*?)\..*/
        return ss1[0][1] + "+" + ss2[0][1]
    }
}

ArrayList values(key) {
    def values = []
    for (level in ['sample', 'bio', 'exp']) {
        def value = key.get(level, null)
        if (value != null) {values << value}
    }
    values << values.join('-')
    return values
}

String tag(key) {
    return values(key)[-1]
}

String catFile(file, t=1) {  
    return file.toString().endsWith('.gz') ? "pbgzip -c -d -n ${t}" : "cat"
}

String catFile1(file, t=1) {  
    return file.toString().endsWith('.gz') ? "gzip -c -d -n ${t}" : "cat"
}

String getName(file) {
    String name
    if (file instanceof String)
        name = file
    else
        name = file.getName()
    return name 
}

Boolean isFastq(file) {
    return (getName(file) ==~ /.*fastq.gz|.*fastq|.*fq|.*fq.gz/)
}

Boolean isSra(file) {
    String name = getName(file)
    return (name ==~ /.*\.sra/) || (name ==~ /^SRR.*/)
}

Boolean isUrl(url) {
    return (url instanceof String) && (
        url ==~ /^https{0,1}:\/\/.*|ftp:\/\//
    )
}

Boolean isFile(file) {
    return file instanceof Path
}

Boolean fileExists(file)
{
    if (file instanceof String)
        file = new File(file)
    return file.exists()
}

/***********************************************digest******************************************/ 

process digest {
    tag {"Digest ${genome.getSimpleName()} with ${enzyme}."}
    publishDir path: {"results/other/frags"}
    memory "2 GB"

    when:
    !frag_path && enzyme 

    input:
    file(fasta) from fasta
    val enzyme_name from enzyme_name
    file(chromsize) from chromsize

    output:
    file("${frag}") into digested_frag

    script:
    frag = "${fasta.getSimpleName()}.${enzyme}.bed"
    """
    cooler digest --out ${frag} ${chromsize} ${fasta} ${enzyme_name}
    """
}
local_frag
    .mix(digested_frag)
    .filter {it ->
        if (do_restrict)
            return it != ""
        else
            return true
    }
    .set {frag_files}


/****************************************load data*********************************/
local_fastqs = Channel.create()
local_sras = Channel.create()
remote_files = Channel.create()

def expanded_raw_reads = params.raw_reads.collect {sample, v1 -> 
    def bios = v1.collect {bio, v2 ->
        def exps = v2.collect{path ->
            if (!isUrl(path) && !isSra(path)) {
                files = file(path, checkIfExists: true)   
                if (isFile(files))
                    return [files]
                else
                    if ((files.size() % 2) != 0)
                        error "Number of files are not even: ${path}"
                    return files.sort().collect { it.toAbsolutePath() }
            }
            else
                return [path]
       }.sum()
       return [bio, exps]
   }
   return [sample, bios]
}
def number_map = [:]
int _id = 0
Channel
    .from(expanded_raw_reads.collect {sample, v1 ->
        number_map["${sample}"] = 0
        v1.collect {bio, v2 ->
            number_map["${sample}"] += 1
            number_map["${sample}-${bio}"] = 0
            Boolean paired = true
            v2.collect {path ->
                number_map["${sample}-${bio}"] += 0.5
                if (isSra(path)) {
                    paired = true
                    number_map["${sample}-${bio}"] += 0.5
                    _id += 1
                } else {
                    if (paired) {_id += 1; paired=false}
                    else paired=true
                }
                key = ['id': _id, 'sample': sample, 'bio': bio]
                return [groupKey(key, 2), path]
            }
        }.sum()
    }.sum())
    .choice(local_fastqs, local_sras, remote_files) {it ->
        def (key, path) = it
        if (isSra(path) && fileExists(path))
            return 1
        else if (fileExists(path))
            return 0
        else
            return 2
    }

remote_files
    .view {it ->
        "Find remote file: ${it[1]}."
    }
    .set {_remote_files}

process download_file {
    tag {"Downloading file ${url} done."}
    publishDir path: {"results/other/downloads"}
    
    input:
    set key, url from _remote_files
    
    output:
    set key, file("*") into downloaded_files
    
    script:
    if (isUrl(url)) {
        """
        wget ${url}
        """
    }
    else
        """
        prefetch -X 90000000000000 -O ./ ${url}
        """
}
downloaded_fastqs = Channel.create()
downloaded_sras =  Channel.create()
other_files = Channel.create()
downloaded_files
    .view {key, file ->
        if (!isFile(file))
            error "Not support multi file downloading within one url."
        return "Downloading ${file.getBaseName()} successed."
    }
    .choice(downloaded_fastqs, downloaded_sras, other_files) {it ->
        def (key, file) = it
        if (isFastq(file))
            return 0
        else if (isSra(file))
            return 1
        else
            return 2
    }

local_sras
    .mix(downloaded_sras)
    .set{sras}

process fastq_dump {
    tag {"Fastq-dump ${sra}"}
    publishDir path: {"results/other/dumped_fastqs"}
    cpus = 10
    memory = "2 GB"

    input:
    set key, file(sra) from sras
    
    output:
    set key, file("*") into dumped_fastqs mode flatten

    script:
    """
    parallel-fastq-dump --sra-id ${sra} --threads ${thread(task)} --gzip --split-files
    """    
}

local_fastqs
    .mix (downloaded_fastqs, dumped_fastqs)
    .map {key, file_path ->
        return [key, (file_path instanceof String) ? file(file_path) : file_path]
    }
    .tap {fastqc_fastqs}
    .groupTuple()
    .map {key, file_pair ->
        def (file1, file2) = file_pair
        def (id, sample, bio) = [key['id'], key['sample'], key['bio']]     
        def exp = commonPrefix(file1.getName(), file2.getName())
        def new_key = ['sample': sample, 'bio': bio, 'exp': exp]
        return [new_key, file1, file2]
    }
    .set {samples}

/***********************************************qc*******************************************/
process fastqc {
    tag {"Fastqc ${fastq}"}
    publishDir path: "results/fastqc/", mode: "copy"
    cpus = 5
    memory = "1 GB"
    when:
    do_fastqc

    input:
    set key, file(fastq) from fastqc_fastqs
    
    output:
    set key, file("*html"), file("*zip") into fastqc_results

    
    script:
    """
    fastqc --threads ${thread(task)} -o `pwd` ${fastq}
    """
}
fastqc_results
    .collect()
    .map {
        return file("results/fastqc")
    }
    .set {fastqc_dir}

process multiqc {
    tag {"Multiqc"}
    publishDir path: "results/fastqc/"

    when:
    do_fastqc 

    input:
    file(fastqc_dir) from fastqc_dir
    
    output:
    set key, file("*") into multiqc_results 
    
    script:
    """
    multiqc ${fastqc_dir}
    """

}

/**************************************split fastq********************************************/
process split_fastq_chunks {
    tag {"Split ${tag(key)}"}
    publishDir path: {"results/other/chunks/${tag(key)}/fastq_chunks"}
    cpus 2

    input:
    set key, fq1, fq2 from samples

    output:
    set key, '*_1*', '*_2*' into _chunk_fastqs

    script:
    (sample, bio, exp, tags) = values(key)
    """
    ${catFile1(fq1)} $fq1 | split -l ${chunksize} -d  - ${exp}_1 &
    ${catFile1(fq2)} $fq2 | split -l ${chunksize} -d  - ${exp}_2 &
    """

}
_chunk_fastqs
    .flatMap{key, fq1s, fq2s ->
        def chunk_list = []
        if (!isFile(fq1s)) {
            def num1 = (fq1s.size() - 1) as int
            def num2 = (fq2s.size() - 1) as int
            if (num1 != num2) {
                error "Number of chunks of two fastq file not equal."
            }   
            for (i in 0..num1) {
                chunk_list << [groupKey(key, num1 + 1), i, fq1s[i], fq2s[i]]  
            }
        }
        else
            chunk_list << [groupKey(key, 1), 0, fq1s, fq2s]
        return chunk_list
    }
    .set {chunk_fastqs}

/*************************************************processing***********************************/
process map_parse_sort {
    tag {"Map and parse ${tag(key)}-${i}"}
    publishDir path: {"results/other/chunks/${tag(key)}/bam_pairs"}
    cpus 10
    memory "10 GB"

    input:
    set key, i, fq1, fq2 from chunk_fastqs
    set file(index_path), index_prefix from index
    file(chromsize) from chromsize

    output:
    set key, file("${bam}"), file("${pair}") into chunk_bam_pairs

    script:
    (sample, bio, exp, tags) = values(key)
    bam = "${exp}_chunk_${i}.bam"
    pair = "${exp}_chunk_${i}.pair.gz"
    keep_bam = merge_bam
        ? "| tee >(samtools sort -O BAM > ${bam})"
        : ""
    t = Math.max((thread(task) - 1)as int, 1)
    """
    touch ${bam}
    bwa mem -SP5M -t ${t} \
        ${index_path}/${index_prefix} ${fq1} ${fq2} ${keep_bam} \
    | pairtools parse \
        --assembly ${assembly} -c ${chromsize} ${parse_args} \
    | pairtools sort --nproc ${t} --tmpdir ${sort_tmpdir} -o ${pair}
    """
}                  
(chunk_bams, chunk_pairs) = \
    chunk_bam_pairs.separate(2) {
        def (key, bam, pair) = it
        [[key, bam], [key, pair]]
    }
chunk_bams
    .groupTuple()
    .set {grouped_chunk_bams}
chunk_pairs
    .groupTuple()
    .set {grouped_chunk_pairs}


process merge_bam {
    tag {"Merge bam chunks of ${tag(key)}"}
    storeDir "results/bams"
    cpus 10
    memory "2 GB"

    when:
    merge_bam

    input:
    set key, file(bams) from grouped_chunk_bams 

    output:
    set new_key, file("${bam}") into exp_bams 

    script:
    (sample, bio, exp, tags) = values(key)
    new_key = ['sample': sample, 'bio': bio]
    bam = "${tags}.bam"
    if (!isFile(bams))
        """
        samtools merge -@ ${thread(task)} -O BAM ${bam} ${bams}  
        """
    else
        """
        cp ${bams} ${bam}
        """
}

process merge_pair {
    tag {"Merge pair chunks of {tag(key)}"}
    storeDir "results/pairs/exp"
    cpus 10
    memory "10 GB"
    
    input:
    set key, file(pairs) from grouped_chunk_pairs
    file(frag_file) from frag_files.first()
    file(chromsize) from chromsize

    output:
    set new_key, file("${pair}") into exp_pairs
    set key, file("${pair}") into exp_pairs_to_cools
    set new_key, file("${stat}") optional true into exp_stats
    set new_key, file("${rest_pair}") into exp_rest_pairs

    script:
    (sample, bio, exp, tags) = values(key)
    new_key = ['sample': sample, 'bio': bio]
    pair = "${tags}.pair.gz"
    stat = "${tags}.raw.stat"
    rest_pair = "${tags}.rest.pair.gz"
    t = Math.max(thread(task) - 2, 1)
    merge_cmd = !isFile(pairs) ? "pairtools merge --nproc ${t}" : "${catFile(pairs, t)}"
    stat_cmd = do_stat ? "| tee >(pairtools stats -o ${stat})" : ""
    restrict_cmd = do_restrict ? " | pairtools restrict -f ${frag_file}" : ""
    if (select_expr)
        select_cmd = """| pairtools select --output-rest add_rest.pair -o ${pair} \"${select_expr}\"
                        pairtools merge -o ${rest_pair} add_rest.pair base_rest.pair
                        rm add_rest.pair base_rest.pair
                     """
    else
        select_cmd = """-o ${pair}
                       pbgzip base_rest.pair && mv base_rest.pair.gz ${rest_pair}
                     """
    """
    ${merge_cmd} ${pairs} ${stat_cmd} \
    | pairtools select --output-rest base_rest.pair --chrom-subset ${chromsize} "True" \
    ${restrict_cmd} ${select_cmd}
    """
}
exp_pairs
    .map {key, exp_pair -> 
            def (sample, bio, tags) = values(key)
            return [groupKey(key, number_map[tags] as int), exp_pair]    
    }
    .groupTuple()
    .set {grouped_exp_pairs}
// TODO: fetch the number of pairs within one sample to incorporate with the groupKey usage.
exp_stats
    .map {key, exp_stat ->
        def sample = key['sample']
        return [['sample': sample], exp_stat]
    }
    .groupTuple()
    .set {grouped_exp_stats}

process merge_exp_dedup {
    tag {"Merge experimental pairs of ${tag(key)}"}
    storeDir "results/pairs/bio"
    cpus 10
    memory "5 GB"

    input:
    set key, file(exp_pairs) from grouped_exp_pairs

    output:
    set new_key, 
        file("${pair}"),
        file("${duplicate}") into bio_pairs
    set key, file("${pair}") into bio_pairs_to_cools

    script:
    (sample, bio, tags) = values(key)
    new_key = ['sample': sample]
    pair = "${tags}.dedup.pair.gz"
    duplicate = "${tags}.dups.pair.gz"
    t = thread(task)
    if (!isFile(exp_pairs))
        merge_cmd = "pairtools merge --nproc ${t}"
    else
        merge_cmd = "${catFile(exp_pairs, t)}"
    """
    ${merge_cmd} ${exp_pairs} | pairtools dedup \
        --max-mismatch 2 --mark-dups \
        --output ${pair} \
        --output-dups ${duplicate} \
    """
}
bio_pairs
    .map {key, bio_pair, duplicate ->
        [key, bio_pair]
    }
    .map {key, bio_pair ->
        def (sample, tags) = values(key)
        [groupKey(key, number_map[tags] as int), bio_pair]
    }.groupTuple()
    .set {grouped_bio_pairs}


process merge_bio {
    tag {"Merge bio pairs of ${tag(key)}"} 
    storeDir "results/pairs/sample"
    cpus 10
    memory "5 GB"

    input:
    set key, file(bio_pairs) from grouped_bio_pairs

    output:
    set key, file("${pair}") into sample_pairs
    set key, file("${pair}") into sample_pairs_to_cools
    script:
    (sample, tags) = values(key)
    pair = "${tags}.pair.gz"
    t = thread(task)
    if (!isFile(bio_pairs))
        """
        pairtools merge --nproc ${t} --output ${pair} ${bio_pairs} 
        """
    else
        """
        cp ${bio_pairs} ${pair}
        """
}
exp_pairs_to_cools
    .mix(bio_pairs_to_cools, sample_pairs_to_cools)
    .set {all_pairs}

process stat_samples {
    tag {"Stat ${tags}"}
    storeDir {"results/pairs/sample"}    

    when:
    do_stat

    input:
    set key, file(pair) from sample_pairs
    
    output:
    set key, file("${stat}") into sample_stats
    
    script:
    tags = values(key)[-1]
    stat = "${tags}.stat"
    """
    pairtools stats --output ${stat} ${pair}
    """
}

process stat_raw_samples {
    tag {"Merge raw stats of ${tags}"} 
    storeDir {"results/pairs/sample"}

    when:
    do_stat

    input:
    set key, file(stats) from grouped_exp_stats
    
    output:
    set key, file(stat) into sample_raw_stats
    
    script:
    tags = values(key)[-1]
    stat = "${tags}.raw.stat"
    """
    pairtools stats --output ${stat} --merge ${stats} 
    """
}

process pair_to_cool {
    tag {" GBenerate minres_cool of ${tag(key)}"} 
    publishDir path: "results/cools/minres"
    memory "4 GB"

    input:
    set key, file(pair) from  all_pairs
    file(chromsize) from chromsize

    output:
    set key, file("${cool}") into minres_cools   

    script:
    tags = values(key)[-1]
    cool = "${tags}_${(min_res / 1000) as int}k.cool"
    """
    cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 \
        --assembly ${assembly} ${chromsize}:${min_res} \
        ${pair} ${cool} 
    """
}
minres_cools
    .filter {key, cool ->
        (key.get('bio', '') == null) && (key.get('exp', '') == null)
    }
    .set {zoom_cools}

process zoomify_cool {
    tag {" GBenerate mcools of ${tag(key)}"} 
    storeDir "results/cools"

    input:
    set key, file(cool) from zoom_cools

    output:
    set key, file("${mcool}") into mcools

    script:
    resolutions = params.resolutions.join(',')
    (sample, tags) = values(key)
    mcool = "${tags}.mcool"
    """
    cooler zoomify --balance --resolutions ${resolutions} --out ${mcool} ${cool}
    """
}

process mcool_to_features {
    tag {"Extract genomic features of ${tag(key)}"} 
    publishDir {"results/features/${tag(key)}"}
    cpus 20
    maxForks 1

    input:
    set key, file(mcool) from mcools
    
    output:
    set key, file("*") into features
    
    script:
    cool_100k = "${mcool}::/resolutions/100000"
    cool_10k = "${mcool}::/resolutions/10000"
    (sample, tags) = values(key)
    """
    hictools compartment ${cool_100k} ${sample}_100k_compartments.bed
    hictools expected ${cool_10k} ${sample}_10k_expected.bed
    hictools peaks call-by-hiccups ${cool_10k} ${sample}_10k_peaks.bed
    hictools tad di-score ${cool_10k} ${sample}_10k_discore.bed
    hictools tad insu-score ${cool_10k} ${sample}_10k_insuscore.bed
    """
}
