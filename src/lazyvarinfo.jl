struct LazyVarInfo{Tvarinfo <: AbstractVarInfo, Tlogp} <: AbstractVarInfo
    varinfo::Tvarinfo
    logp::Tlogp
    lastidx::Base.RefValue{Int}
end
LazyVarInfo(vi::AbstractVarInfo) = LazyVarInfo(vi, [vi.logp[]], Ref(1))
LazyVarInfo(m::Model) = LazyVarInfo(VarInfo(m))

TypedVarInfo(vi::LazyVarInfo) = LazyVarInfo(TypedVarInfo(vi.varinfo), vi.logp, vi.lastidx)
function VarInfo(old_vi::LazyVarInfo, spl, x::AbstractVector)
    vi = VarInfo(old_vi.varinfo, spl, x)
    T = typeof(vi.logp[])
    return LazyVarInfo(vi, convert(Vector{T}, old_vi.logp), old_vi.lastidx)
end

function getlogp(vi::LazyVarInfo)
    (length(vi.logp) == 0 || vi.lastidx[] == 0) && return zero(eltype(vi.logp))
    return sum(view(vi.logp, 1:vi.lastidx[]))
end
function resetlogp!(vi::LazyVarInfo)
    vi.lastidx[] = 0
    return zero(eltype(vi.logp))
end
getlogpvec(vi::LazyVarInfo) = vi.logp
function acclogp!(vi::LazyVarInfo, logp::Real)
    if length(vi.logp) == vi.lastidx[]
        push!(vi.logp, logp)
    else
        idx = (vi.lastidx[] += 1)
        vi.logp[idx] = logp
    end
    return logp
end
function setlogp!(vi::LazyVarInfo, logp::Real)
    if length(vi.logp) == 0
        push!(vi.logp, logp)
    else
        vi.logp[1] = logp
    end
    vi.lastidx[] = 1
    return logp
end

get_num_produce(vi::LazyVarInfo) = get_num_produce(vi.varinfo)
increment_num_produce!(vi::LazyVarInfo) = increment_num_produce!(vi.varinfo)
reset_num_produce!(vi::LazyVarInfo) = reset_num_produce!(vi.varinfo)
set_num_produce!(vi::LazyVarInfo, n::Int) = set_num_produce!(vi.varinfo, n)

getall(vi::LazyVarInfo) = getall(vi.varinfo)
syms(vi::LazyVarInfo) = syms(vi.varinfo)
isempty(vi::LazyVarInfo) = isempty(vi.varinfo)
getmetadata(vi::LazyVarInfo, vn::VarName) = getmetadata(vi.varinfo, vn)
getidx(vi::LazyVarInfo, vn::VarName) = getidx(vi.varinfo, vn)
getrange(vi::LazyVarInfo, vn::VarName) = getrange(vi.varinfo, vn)
keys(vi::LazyVarInfo) = keys(vi.varinfo)
haskey(vi::LazyVarInfo, vn::VarName) = haskey(vi.varinfo, vn)
tonamedtuple(vi::LazyVarInfo) = tonamedtuple(vi.varinfo)
setall!(vi::LazyVarInfo, val) = setall!(vi.varinfo, val)
_getranges(vi::LazyVarInfo, idcs::NamedTuple) = _getranges(vi.varinfo, idcs)
_getidcs(vi::LazyVarInfo, spl::SampleFromPrior) = _getidcs(vi.varinfo, spl)
_getidcs(vi::LazyVarInfo, s::Selector, space) = _getidcs(vi.varinfo, s, space)
_getvns(vi::LazyVarInfo, spl::SampleFromPrior) = _getvns(vi.varinfo, spl)
_getvns(vi::LazyVarInfo, s::Selector, space) = _getvns(vi.varinfo, s, space)
link!(vi::LazyVarInfo, spl::AbstractSampler) = link!(vi.varinfo, spl)
invlink!(vi::LazyVarInfo, spl::AbstractSampler) = invlink!(vi.varinfo, spl)
islinked(vi::LazyVarInfo, spl::AbstractSampler) = islinked(vi.varinfo, spl)
getindex(vi::LazyVarInfo, spl::Sampler) = getindex(vi.varinfo, spl)
setindex!(vi::LazyVarInfo, val, spl::Sampler) = setindex!(vi.varinfo, val, spl)
set_retained_vns_del_by_spl!(vi::LazyVarInfo, spl::Sampler) = set_retained_vns_del_by_spl!(vi.varinfo, spl)
function empty!(vi::LazyVarInfo)
    empty!(vi.varinfo)
    resetlogp!(vi)
    return vi
end
push_assert(vi::LazyVarInfo, vn::VarName, dist, gidset) = push_assert(vi.varinfo, vn, dist, gidset)
