with actuals as (
    select 'Canada' as Country, convert(datetime, [File Date], 112) as Date, [Ship to party], [Material], sum([3P Sales Qty Total MT]) as [3P Sales Qty Total MT],
           sum([N3P Net Revenue]) as [N3P Net Revenue], sum([New COGS]) as [New COGS], sum([Operating Income]) as [Operating Income]
    from dbo.pl_data_can_actuals
    group by [File Date], [Ship to party], [Material]
    union all
    select 'Mexico' as Country, convert(datetime, [File Date], 112) as Date, [Ship to party], [Material], sum([3P Sales Qty Total MT]) as [3P Sales Qty Total MT],
           sum([N3P Net Revenue]) as [N3P Net Revenue], sum([New COGS]) as [New COGS], sum([Operating Income]) as [Operating Income]
    from dbo.pl_data_mex_actuals
    group by [File Date], [Ship to party], [Material]
    union all
    select 'USA' as Country, convert(datetime, [File Date], 112) as Date, [Ship to party], [Material], sum([3P Sales Qty Total MT]) as [3P Sales Qty Total MT],
           sum([N3P Net Revenue]) as [N3P Net Revenue], sum([New COGS]) as [New COGS], sum([Operating Income]) as [Operating Income]
    from dbo.pl_data_us_actuals
    group by [File Date], [Ship to party], [Material]
),
     actuals_piv as (
         select Country, [Date], [Ship to party], Component, Actuals, Material
         from actuals
                  unpivot (
                  Actuals for Component in
                 ([3P Sales Qty Total MT],[N3P Net Revenue],[New COGS], [Operating Income])
                  ) as actuals_piv
     ),
     le0 as (
         select 'Canada' as Country, convert(datetime, [File Date], 112) as Date, [Ship to party], [Material], sum([3P Sales Qty Total MT]) as [3P Sales Qty Total MT],
                sum([N3P Net Revenue]) as [N3P Net Revenue], sum([New COGS]) as [New COGS], sum([Operating Income]) as [Operating Income]
         from dbo.pl_data_can_le0
         group by [File Date], [Ship to party], [Material]
         union all
         select 'Mexico' as Country, convert(datetime, [File Date],112) as Date, [Ship to party], [Material], sum([3P Sales Qty Total MT]) as [3P Sales Qty Total MT],
                sum([N3P Net Revenue]) as [N3P Net Revenue], sum([New COGS]) as [New COGS], sum([Operating Income]) as [Operating Income]
         from dbo.pl_data_mex_le0
         group by [File Date], [Ship to party], [Material]
         union all
         select 'USA' as Country, convert(datetime, [File Date], 112) as Date, [Ship to party], [Material], sum([3P Sales Qty Total MT]) as [3P Sales Qty Total MT],
                sum([N3P Net Revenue]) as [N3P Net Revenue], sum([New COGS]) as [New COGS], sum([Operating Income]) as [Operating Income]
         from dbo.pl_data_us_le0
         group by [File Date], [Ship to party], [Material]
     ),
     le0s_piv as (
         select Country, [Date], [Ship to party], Component, le0, Material
         from le0
                  unpivot (
                  le0 for Component in
                 ([3P Sales Qty Total MT],[N3P Net Revenue],[New COGS],[Operating Income])
                  ) as le0s_piv
     )
select actuals_piv.Country, actuals_piv.[Date], actuals_piv.[Ship to party], actuals_piv.Material, actuals_piv.Component, Actuals, le0
from actuals_piv
left join le0s_piv on(coalesce(actuals_piv.Country,'')=coalesce(le0s_piv.Country, '-') and coalesce(actuals_piv.[Date],'-')=coalesce(le0s_piv.[Date],'-') and
                   coalesce(actuals_piv.[Ship to party],'-')=coalesce(le0s_piv.[Ship to party],'-') and coalesce(actuals_piv.Material,'-')=coalesce(le0s_piv.Material,'-') and
                   coalesce(actuals_piv.Component,'-')=coalesce(le0s_piv.Component,'-'))
where actuals_piv.[Date] >= '20180101'
union all
select le0s_piv.Country, le0s_piv.[Date], le0s_piv.[Ship to party], le0s_piv.Material, le0s_piv.Component, Actuals, le0
from actuals_piv
         right join le0s_piv on(actuals_piv.Country=le0s_piv.Country and actuals_piv.[Date]=le0s_piv.[Date] and
                                actuals_piv.[Ship to party]=le0s_piv.[Ship to party] and actuals_piv.Material=le0s_piv.Material and actuals_piv.Component=le0s_piv.Component)
where  actuals_piv.Country is null and le0s_piv.[Date] >= '20180101'